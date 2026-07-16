<script setup lang="ts">
defineProps<{
  progress: number
  label?: string
}>()
</script>

<template>
  <div class="upload-progress">
    <div class="row">
      <span class="label">{{ label ?? '正在上传视频…' }}</span>
      <span class="percent">{{ progress }}%</span>
    </div>
    <div class="track">
      <div class="bar" :style="{ width: `${progress}%` }" />
    </div>
  </div>
</template>

<style scoped>
.upload-progress {
  padding: 1rem;
  border: 1px solid var(--line);
  background: rgba(0, 0, 0, 0.2);
}

.row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.6rem;
}

.label {
  font-size: 0.9rem;
  color: var(--text);
}

.percent {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: var(--marking);
}

.track {
  height: 10px;
  border-radius: 999px;
  background: rgba(139, 148, 158, 0.25);
  overflow: hidden;
}

.bar {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--marking), var(--accent));
  transition: width 0.15s ease;
}
</style>
