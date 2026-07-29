<script setup lang="ts">
import { computed } from 'vue'
import type { EnvProb } from '@/api/environment'

const props = defineProps<{ env: EnvProb | null }>()

function probList(probs: Record<string, number> | null | undefined) {
  if (!probs) return []
  return Object.entries(probs).map(([label, value]) => ({ label, value }))
}

// VLM-backed results all share the product name regardless of the concrete
// backend model; heuristics keep their raw id (e.g. heuristic-v1) so the
// fallback path stays distinguishable.
const displayModel = computed(() => {
  const model = props.env?.model ?? ''
  return model.startsWith('vlm:') ? 'IntelliVisionEnv' : model
})
</script>

<template>
  <div v-if="env" class="env-bars">
    <div class="env-dim-row">
      <span class="env-tag">天气</span><strong>{{ env.weather }}</strong>
    </div>
    <div v-for="p in probList(env.weather_probs)" :key="'w' + p.label" class="prob">
      <span class="prob-label">{{ p.label }}</span>
      <span class="prob-track"><i class="prob-fill" :style="{ width: (p.value * 100).toFixed(0) + '%' }" /></span>
      <span class="prob-val">{{ (p.value * 100).toFixed(0) }}%</span>
    </div>

    <div class="env-dim-row">
      <span class="env-tag">路面</span><strong>{{ env.road_condition }}</strong>
    </div>
    <div v-for="p in probList(env.road_probs)" :key="'r' + p.label" class="prob">
      <span class="prob-label">{{ p.label }}</span>
      <span class="prob-track"><i class="prob-fill" :style="{ width: (p.value * 100).toFixed(0) + '%' }" /></span>
      <span class="prob-val">{{ (p.value * 100).toFixed(0) }}%</span>
    </div>

    <div class="env-dim-row">
      <span class="env-tag">能见度</span><strong>{{ env.visibility }}</strong>
    </div>
    <div v-for="p in probList(env.visibility_probs)" :key="'v' + p.label" class="prob">
      <span class="prob-label">{{ p.label }}</span>
      <span class="prob-track"><i class="prob-fill" :style="{ width: (p.value * 100).toFixed(0) + '%' }" /></span>
      <span class="prob-val">{{ (p.value * 100).toFixed(0) }}%</span>
    </div>

    <div class="env-dim-row">
      <span class="env-tag">交通</span><strong>{{ env.traffic_condition || '—' }}</strong>
    </div>

    <p class="env-desc">
      <span v-if="env.description">{{ env.description }}</span><span
        v-if="env.has_accident && env.accident_desc"
        class="env-accident"
      >⚠ 交通事故：{{ env.accident_desc }}</span>
    </p>
    <p v-if="env.is_night" class="env-night">夜间场景</p>
    <p class="env-model">识别模型：{{ displayModel }}</p>
  </div>
</template>

<style scoped>
.env-bars {
  display: grid;
  gap: 0.3rem;
}

.env-dim-row {
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
  margin-top: 0.4rem;
}

.env-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  min-width: 3rem;
}

.env-dim-row strong {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.1rem;
  color: var(--marking);
}

.prob {
  display: grid;
  grid-template-columns: 3rem 1fr 2.5rem;
  align-items: center;
  gap: 0.4rem;
}

.prob-label {
  font-size: 0.78rem;
  color: var(--text-muted);
}

.prob-track {
  height: 0.5rem;
  background: rgba(139, 148, 158, 0.18);
  border: 1px solid rgba(139, 148, 158, 0.22);
  overflow: hidden;
}

.prob-fill {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, var(--marking), var(--accent));
}

.prob-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.74rem;
  text-align: right;
  color: var(--text);
}

.env-night {
  margin: 0.35rem 0 0;
  font-size: 0.78rem;
  color: var(--marking);
}

.env-desc {
  margin: 0.35rem 0 0;
  font-size: 0.78rem;
  color: var(--text);
  line-height: 1.5;
}

.env-accident {
  margin-left: 0.4rem;
  padding: 0.05rem 0.4rem;
  border-radius: 4px;
  background: rgba(255, 77, 79, 0.12);
  border: 1px solid rgba(255, 77, 79, 0.5);
  color: #ff6b6b;
  font-size: 0.78rem;
  font-weight: 600;
  white-space: normal;
}

.env-model {
  margin: 0.25rem 0 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  color: var(--text-muted);
}
</style>
