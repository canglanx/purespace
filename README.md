# PureSpace: A Benchmark for Abstract Spatial Reasoning in Vision-Language Models

## How to Generate Your Own Data

**1. Clone repository**
```bash
git clone https://github.com/canglanx/purespace.git
cd purespace
```

**2. Setup environment**
```bash
uv sync
```

**3. Configure output directory**
```bash
nano ./configs/data_generation/demo.yaml
```

**4. Run quick demo**
```bash
uv run python -m purespace.data_generation.cli --config ./configs/data_generation/demo.yaml
```

**5. (Alternatively) Run large-scale generation with debug mode**
```bash
uv run python -m purespace.data_generation.cli --config ./configs/data_generation/large.yaml --debug
```
