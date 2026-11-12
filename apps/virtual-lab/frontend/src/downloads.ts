export interface DownloadState {
  phase: string;
  file?: string;
  downloaded_bytes?: number;
  total_bytes?: number | null;
  files_completed?: number;
  active_files?: number;
}
function bytes(value: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB']; let index = 0;
  while (value >= 1000 && index < units.length - 1) { value /= 1000; index++; }
  return `${value.toFixed(index ? 1 : 0)} ${units[index]}`;
}
export function downloadView(download: DownloadState | null | undefined, phase: string) {
  const received = Math.max(0, download?.downloaded_bytes ?? 0), total = download?.total_bytes;
  const known = typeof total === 'number' && Number.isFinite(total) && total > 0;
  const value = known ? Math.min(received, total) : undefined;
  const label = phase === 'load_failed' || phase === 'failed' || download?.phase === 'failed' ? 'Model load failed' :
    download?.phase === 'downloading' ? 'Downloading model files' : download?.phase === 'loading' ? 'Loading model' : 'Preparing model';
  const detail = download?.file ? `${download.file} · ${bytes(received)}${known ? ` / ${bytes(total)} · ${Math.floor(value! / total * 100)}% of this file` : ' · file size unknown'}` :
    'Checking cached files and preparing the subject.';
  const files = download?.files_completed ?? 0, active = download?.active_files ?? 0;
  return { visible: phase === 'loading' || Boolean(download), label,
    detail: `${detail}${files ? ` · ${files} file${files === 1 ? '' : 's'} complete` : ''}${active > 1 ? ` · ${active} active files` : ''}`,
    value, total: known ? total : 1 };
}
export function renderDownload(download: DownloadState | null | undefined, phase: string): void {
  const view = downloadView(download, phase);
  document.getElementById('model-download-progress')!.hidden = !view.visible;
  document.getElementById('model-download-state')!.textContent = view.label;
  document.getElementById('model-download-detail')!.textContent = view.detail;
  const bar = document.getElementById('model-download-bar') as HTMLProgressElement;
  bar.max = view.total;
  if (view.value === undefined) bar.removeAttribute('value'); else bar.value = view.value;
}
