use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};
use std::io::Write;

use burn::{
    optim::AdamWConfig,
    tensor::{Data, ElementConversion, Shape, Tensor, Int},
    prelude::*,
    record::DefaultRecorder,
    train::{
        ClassificationOutput,
        TrainStep,
        TrainOutput,
        LearnerBuilder,
        ValidStep,
        logger::MetricLogger,
        metric::{Adaptor, LossMetric, LossInput, Metric, MetricEntry, MetricMetadata},
    },
};
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::backend::AutodiffBackend as AutodiffBackendTrait;
use burn_wgpu::{AutoGraphicsApi, Wgpu, WgpuDevice};
use wgpu::DeviceType;

use crate::config::ModelConfig;
use crate::data::{books::read_books_folder, tokenizer::ByteTokenizer, dataset::make_batch};
use crate::model::gpt::Gpt;
use crate::checkpoint::save_config;
use crate::settings::{AppSettings, EarlyStopSettings};

type BackendImpl = burn_ndarray::NdArray<f32>;
type AutodiffBackendImpl = burn::backend::Autodiff<BackendImpl>;
type WgpuBackend = Wgpu<AutoGraphicsApi, f32, i32>;
type WgpuAutodiffBackend = burn::backend::Autodiff<WgpuBackend>;

#[derive(Module, Debug)]
pub struct GptTrainingModel<B: Backend> {
    pub gpt: Gpt<B>,
}

impl<B: Backend> GptTrainingModel<B> {
    // Build a training model from a config on the given device.
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        Self { gpt: Gpt::new(cfg, device) }
    }
    // Forward pass that returns logits for loss computation.
    pub fn forward_logits(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.gpt.forward(x)
    }
}

#[derive(Clone, Debug)]
pub struct TrainBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

#[derive(Clone)]
struct TrainBatcher<B: Backend> {
    device: B::Device,
    seq_len: usize,
}

impl<B: Backend> TrainBatcher<B> {
    // Create a batcher that reshapes flat token buffers into tensors.
    fn new(device: B::Device, seq_len: usize) -> Self {
        Self { device, seq_len }
    }
}

impl<B: Backend> Batcher<(Vec<u32>, Vec<u32>), TrainBatch<B>> for TrainBatcher<B> {
    // Convert raw token vectors into input/target tensors.
    fn batch(&self, items: Vec<(Vec<u32>, Vec<u32>)>) -> TrainBatch<B> {
        let mut inputs_flat = Vec::new();
        let mut targets_flat = Vec::new();
        for (inputs, targets) in items {
            inputs_flat.extend(inputs);
            targets_flat.extend(targets);
        }
        let total_tokens = inputs_flat.len();
        assert!(total_tokens % self.seq_len == 0);
        let batch = total_tokens / self.seq_len;

        let inputs = Tensor::<B, 2, Int>::from_data(
            Data::new(
                inputs_flat
                    .iter()
                    .map(|&v| v.elem::<B::IntElem>())
                    .collect::<Vec<_>>(),
                Shape::new([batch, self.seq_len]),
            ),
            &self.device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            Data::new(
                targets_flat
                    .iter()
                    .map(|&v| v.elem::<B::IntElem>())
                    .collect::<Vec<_>>(),
                Shape::new([batch, self.seq_len]),
            ),
            &self.device,
        );
        TrainBatch { inputs, targets }
    }
}

struct TrainOutputItem<B: Backend> {
    output: ClassificationOutput<B>,
}

impl<B: Backend> Adaptor<LossInput<B>> for TrainOutputItem<B> {
    // Adapt the wrapped output to a loss metric input.
    fn adapt(&self) -> LossInput<B> {
        self.output.adapt()
    }
}

impl<B: AutodiffBackendTrait> TrainStep<TrainBatch<B>, TrainOutputItem<B>> for GptTrainingModel<B> {
    // Training step that computes loss and gradients.
    fn step(&self, batch: TrainBatch<B>) -> TrainOutput<TrainOutputItem<B>> {
        let logits = self.forward_logits(batch.inputs);
        let [b, t, v] = logits.dims();
        let logits2 = logits.reshape([b * t, v]);
        let targets = batch.targets.reshape([b * t]);
        let device = logits2.device();
        let loss = burn::nn::loss::CrossEntropyLoss::new(None, &device)
            .forward(logits2.clone(), targets.clone());
        let loss_val = loss.clone().into_data().value[0].elem::<f32>();
        log_loss_sample(loss_val);
        let grads = loss.backward();
        let output = ClassificationOutput::new(loss.clone(), logits2, targets);
        TrainOutput::new(self, grads, TrainOutputItem { output })
    }
}

impl<B: Backend> ValidStep<TrainBatch<B>, TrainOutputItem<B>> for GptTrainingModel<B> {
    // Validation step that computes loss without gradients.
    fn step(&self, batch: TrainBatch<B>) -> TrainOutputItem<B> {
        let logits = self.forward_logits(batch.inputs);
        let [b, t, v] = logits.dims();
        let logits2 = logits.reshape([b * t, v]);
        let targets = batch.targets.reshape([b * t]);
        let device = logits2.device();
        let loss = burn::nn::loss::CrossEntropyLoss::new(None, &device)
            .forward(logits2.clone(), targets.clone());
        let output = ClassificationOutput::new(loss, logits2, targets);
        TrainOutputItem { output }
    }
}

#[derive(Clone)]
struct BackendInfoMetric {
    label: String,
}

impl BackendInfoMetric {
    // Create a metric that reports the selected backend label.
    fn new(label: &str) -> Self {
        Self { label: label.to_string() }
    }
}

impl Metric for BackendInfoMetric {
    const NAME: &'static str = "Backend";
    type Input = ();

    // Emit the backend label as a metric entry.
    fn update(&mut self, _item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        MetricEntry::new(
            Self::NAME.to_string(),
            self.label.clone(),
            self.label.clone(),
        )
    }

    // No internal state to reset between epochs.
    fn clear(&mut self) {}
}

struct StepLimitMetric {
    interrupter: burn::train::TrainingInterrupter,
    max_steps: usize,
    current: usize,
}

impl StepLimitMetric {
    fn new(interrupter: burn::train::TrainingInterrupter, max_steps: usize) -> Self {
        Self { interrupter, max_steps, current: 0 }
    }
}

impl Metric for StepLimitMetric {
    const NAME: &'static str = "Steps";
    type Input = ();

    fn update(&mut self, _item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        self.current += 1;
        if self.current >= self.max_steps {
            self.interrupter.stop();
        }
        let formatted = format!("{}/{}", self.current, self.max_steps);
        MetricEntry::new(Self::NAME.to_string(), formatted.clone(), formatted)
    }

    fn clear(&mut self) {}
}

#[derive(Clone)]
struct NoopMetricLogger;

impl MetricLogger for NoopMetricLogger {
    // Ignore metric updates.
    fn log(&mut self, _item: &MetricEntry) {}
    // Ignore epoch boundary.
    fn end_epoch(&mut self, _epoch: usize) {}
    // No numeric logs available.
    fn read_numeric(&mut self, _name: &str, _epoch: usize) -> Result<Vec<burn::train::metric::NumericEntry>, String> {
        Ok(Vec::new())
    }
}

struct MultiMetricLogger {
    loggers: Vec<Box<dyn MetricLogger>>,
}

impl MultiMetricLogger {
    fn new(loggers: Vec<Box<dyn MetricLogger>>) -> Self {
        Self { loggers }
    }
}

impl MetricLogger for MultiMetricLogger {
    fn log(&mut self, item: &MetricEntry) {
        for logger in &mut self.loggers {
            logger.log(item);
        }
    }

    fn end_epoch(&mut self, epoch: usize) {
        for logger in &mut self.loggers {
            logger.end_epoch(epoch);
        }
    }

    fn read_numeric(&mut self, name: &str, epoch: usize) -> Result<Vec<burn::train::metric::NumericEntry>, String> {
        let mut out = Vec::new();
        for logger in &mut self.loggers {
            let mut vals = logger.read_numeric(name, epoch)?;
            out.append(&mut vals);
        }
        Ok(out)
    }
}

fn log_loss_sample(value: f32) {
    let Ok(path) = std::env::var("BOOKHEAD_LOSS_LOG") else { return };
    if let Some(parent) = std::path::Path::new(&path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{value}");
    }
}

#[derive(Clone)]
struct EarlyStopLogger {
    interrupter: burn::train::TrainingInterrupter,
    config: EarlyStopSettings,
    best: Option<f64>,
    bad_epochs: usize,
    last_loss: Option<f64>,
}

impl EarlyStopLogger {
    // Create a logger that stops training based on loss.
    fn new(interrupter: burn::train::TrainingInterrupter, config: EarlyStopSettings) -> Self {
        Self {
            interrupter,
            config,
            best: None,
            bad_epochs: 0,
            last_loss: None,
        }
    }
}

impl MetricLogger for EarlyStopLogger {
    // Capture the latest loss metric value.
    fn log(&mut self, item: &MetricEntry) {
        if item.name != "Loss" {
            return;
        }
        if let Ok(val) = item.serialize.parse::<f64>() {
            self.last_loss = Some(val);
        }
    }

    // Evaluate patience at the end of each epoch.
    fn end_epoch(&mut self, _epoch: usize) {
        let Some(loss) = self.last_loss else { return };
        match self.best {
            None => {
                self.best = Some(loss);
                self.bad_epochs = 0;
            }
            Some(best) => {
                if loss < best - self.config.min_delta {
                    self.best = Some(loss);
                    self.bad_epochs = 0;
                } else {
                    self.bad_epochs += 1;
                    if self.bad_epochs >= self.config.patience {
                        self.interrupter.stop();
                    }
                }
            }
        }
    }

    // No numeric logs exposed.
    fn read_numeric(&mut self, _name: &str, _epoch: usize) -> Result<Vec<burn::train::metric::NumericEntry>, String> {
        Ok(Vec::new())
    }
}

// Entry point for training with auto backend selection.
pub fn run_train(
    books_dir: &str,
    out_dir: &str,
    seq_len: usize,
    batch_size: usize,
    steps: usize,
    lr: f64,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    d_ff: usize,
    _save_every: usize,
    workers: usize,
    settings_path: Option<&str>,
) -> Result<()> {
    let settings = match settings_path {
        Some(path) => Some(AppSettings::load(path)?),
        None => None,
    };
    let early_stop = settings
        .as_ref()
        .and_then(|s| s.training.as_ref())
        .and_then(|t| t.early_stop.clone())
        .filter(|cfg| cfg.enabled);

    match select_training_backend() {
        TrainingBackend::Gpu { device, label } => run_train_with_backend::<WgpuAutodiffBackend>(
            device,
            &format!("wgpu GPU ({label})"),
            books_dir,
            out_dir,
            seq_len,
            batch_size,
            steps,
            lr,
            d_model,
            n_layers,
            n_heads,
            d_ff,
            workers,
            early_stop.clone(),
        ),
        TrainingBackend::Cpu => run_train_with_backend::<AutodiffBackendImpl>(
            burn_ndarray::NdArrayDevice::default(),
            "ndarray CPU",
            books_dir,
            out_dir,
            seq_len,
            batch_size,
            steps,
            lr,
            d_model,
            n_layers,
            n_heads,
            d_ff,
            workers,
            early_stop,
        ),
    }
}

enum TrainingBackend {
    Cpu,
    Gpu { device: WgpuDevice, label: String },
}

// Pick the best available device and label it for UI.
fn select_training_backend() -> TrainingBackend {
    let device = WgpuDevice::BestAvailable;
    let info = pollster::block_on(burn_wgpu::select_device::<AutoGraphicsApi>(&device)).2;
    let device_type = info.device_type;
    let label = format!("{:?}: {}", device_type, info.name);

    match device_type {
        DeviceType::Cpu => TrainingBackend::Cpu,
        _ => TrainingBackend::Gpu { device, label },
    }
}

fn run_train_with_backend<B: AutodiffBackendTrait>(
    device: B::Device,
    backend_label: &str,
    books_dir: &str,
    out_dir: &str,
    seq_len: usize,
    batch_size: usize,
    steps: usize,
    lr: f64,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    d_ff: usize,
    workers: usize,
    early_stop: Option<EarlyStopSettings>,
) -> Result<()>
where
    B::Device: Clone,
{
    // Train on the selected backend and show it in the UI.
    std::fs::create_dir_all(out_dir)?;
    println!("Training backend: {backend_label}");

    // combine all books into one stream
    let books = read_books_folder(books_dir)?;
    let mut combined = String::new();
    for (_name, text) in books {
        combined.push_str(&text);
        combined.push_str("\n\n");
    }
    let tokens = ByteTokenizer::encode(&combined);

    let cfg = ModelConfig { vocab: 256, seq_len, n_layers, n_heads, d_model, d_ff, dropout: 0.0 };
    save_config(out_dir, &cfg)?;

    let model = GptTrainingModel::<B>::new(&cfg, &device);

    // Pre-generate random batches (keeps code simple).
    let mut rng = StdRng::seed_from_u64(42);
    let pregen = 200usize;
    let mut batches: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(pregen);
    for _ in 0..pregen {
        batches.push(make_batch(&tokens, seq_len, batch_size, &mut rng));
    }

    #[derive(Clone)]
    struct SimpleDataset {
        batches: Vec<(Vec<u32>, Vec<u32>)>,
    }

    impl burn::data::dataset::Dataset<(Vec<u32>, Vec<u32>)> for SimpleDataset {
        // Retrieve a pre-generated batch by index.
        fn get(&self, index: usize) -> Option<(Vec<u32>, Vec<u32>)> { self.batches.get(index).cloned() }
        // Total number of available batches.
        fn len(&self) -> usize { self.batches.len() }
    }

    let dataset = SimpleDataset { batches };

    let batcher = TrainBatcher::<B>::new(device.clone(), seq_len);
    let dataloader = burn::data::dataloader::DataLoaderBuilder::new(batcher)
        .batch_size(1)
        .num_workers(workers.max(1))
        .build(dataset.clone());

    let batcher_valid = TrainBatcher::<B::InnerBackend>::new(device.clone(), seq_len);
    let dataloader_valid = burn::data::dataloader::DataLoaderBuilder::new(batcher_valid)
        .batch_size(1)
        .num_workers(workers.max(1))
        .build(dataset);

    let optim = AdamWConfig::new().init();

    let backend_metric = BackendInfoMetric::new(backend_label);
    let mut learner_builder = LearnerBuilder::new(out_dir)
        .metric_train(LossMetric::new())
        .metric_train(backend_metric.clone())
        .metric_valid(backend_metric)
        .with_file_checkpointer(DefaultRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs((steps + (pregen - 1)) / pregen);

    let step_interrupter = learner_builder.interrupter();
    learner_builder = learner_builder.metric_train(StepLimitMetric::new(step_interrupter, steps));

    let mut train_loggers: Vec<Box<dyn MetricLogger>> = Vec::new();
    let mut valid_loggers: Vec<Box<dyn MetricLogger>> = Vec::new();

    if let Some(cfg) = &early_stop {
        if cfg.use_validation {
            learner_builder = learner_builder.metric_valid(LossMetric::new());
        }
        let interrupter = learner_builder.interrupter();
        let logger = EarlyStopLogger::new(interrupter, cfg.clone());
        if cfg.use_validation {
            valid_loggers.push(Box::new(logger));
        } else {
            train_loggers.push(Box::new(logger));
        }
    }

    if !train_loggers.is_empty() || !valid_loggers.is_empty() {
        if train_loggers.is_empty() {
            train_loggers.push(Box::new(NoopMetricLogger));
        }
        if valid_loggers.is_empty() {
            valid_loggers.push(Box::new(NoopMetricLogger));
        }
        learner_builder = learner_builder.metric_loggers(
            MultiMetricLogger::new(train_loggers),
            MultiMetricLogger::new(valid_loggers),
        );
    }

    let learner = learner_builder.build(model, optim, lr);

    learner.fit(dataloader, dataloader_valid);

    Ok(())
}
