# Human coach schedules

Handcrafted schedules authored by a human coach for the `human_coach` scenario of the Progressive Healthy Lifestyle Challenge. Each `<person_id>.json` is one person's scheduling solution. The schedules are provided as input and scored like any other method. To reproduce the evaluation, copy these files into the run's `augmented/persons/` directory and run `evaluate`:

```bash
RUN=output/progressive_healthy_lifestyle_promotion/scenarios/human_coach/human_coach
mkdir -p "$RUN/augmented/persons"
cp src/experiments/persona/healthy_lifestyle_promotion/human_coach_schedules/*.json \
    "$RUN/augmented/persons/"

sudo docker compose -f docker-compose.gpu.yml run --rm -e COMPUTE_DEVICE=cuda app \
    python -m src.scripts.scenarios.cli evaluate \
        --scenario src/experiments/persona/healthy_lifestyle_promotion/scenarios.yaml \
        --scenario-id human_coach --learning-charts
```
