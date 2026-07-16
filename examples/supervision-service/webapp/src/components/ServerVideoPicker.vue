<script setup lang="ts">
import { onMounted, ref } from 'vue'

import { fetchServerVideos, type ServerVideo } from '@/api/video'

const emit = defineEmits<{
  select: [video: ServerVideo]
}>()

const videos = ref<ServerVideo[]>([])
const loading = ref(false)
const errorMessage = ref('')

async function load() {
  loading.value = true
  errorMessage.value = ''
  try {
    videos.value = await fetchServerVideos()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '加载示例视频失败。'
  } finally {
    loading.value = false
  }
}

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(0)} KB`
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

onMounted(load)
</script>

<template>
  <div class="server-videos">
    <div class="header">
      <h3>示例视频</h3>
      <button type="button" class="refresh" :disabled="loading" @click="load">
        {{ loading ? '刷新中…' : '刷新' }}
      </button>
    </div>

    <p v-if="loading" class="muted">加载中…</p>
    <p v-else-if="errorMessage" class="error">{{ errorMessage }}</p>
    <p v-else-if="!videos.length" class="muted">服务端暂无可用的示例视频。</p>

    <ul v-else class="list">
      <li v-for="video in videos" :key="video.name">
        <button type="button" class="item" @click="emit('select', video)">
          <span class="name" :title="video.name">{{ video.name }}</span>
          <span class="size">{{ formatSize(video.size) }}</span>
        </button>
      </li>
    </ul>
  </div>
</template>

<style scoped>
.server-videos {
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 0.85rem;
  background: rgba(0, 0, 0, 0.18);
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.6rem;
}

.header h3 {
  margin: 0;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.98rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.refresh {
  padding: 0.3rem 0.6rem;
  border: 1px solid var(--line);
  background: transparent;
  color: var(--marking);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  cursor: pointer;
}

.refresh:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.4rem;
  max-height: 220px;
  overflow-y: auto;
}

.item {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.75rem;
  padding: 0.5rem 0.65rem;
  border: 1px solid rgba(139, 148, 158, 0.35);
  background: var(--bg-base);
  color: var(--text);
  cursor: pointer;
  text-align: left;
}

.item:hover {
  border-color: rgba(245, 197, 24, 0.6);
  background: rgba(245, 197, 24, 0.06);
}

.name {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 0.82rem;
}

.size {
  flex-shrink: 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: var(--text-muted);
}

.muted {
  margin: 0;
  color: var(--text-muted);
  font-size: 0.82rem;
}

.error {
  margin: 0;
  color: var(--danger);
  font-size: 0.82rem;
}
</style>
