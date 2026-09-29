<script lang="ts">
	import { onMount } from 'svelte';
	import { SvelteSet } from 'svelte/reactivity';
	import { resolve } from '$app/paths';
	import { LineChart } from 'layerchart';

	const FOLDERS_URL = 'http://127.0.0.1:5290/data-folders';
	const PREVIEW_URL = 'http://127.0.0.1:5290/data-folders/preview';
	const RUN_URL = 'http://127.0.0.1:5290/postprocess';

	let folders = $state<string[]>([]);
	let loadError = $state<string | null>(null);
	const selected = new SvelteSet<string>();

	let points = $state<{ time: Date; water_content: number }[]>([]);
	let previewState = $state<'idle' | 'loading' | 'error'>('idle');
	let previewMessage = $state('');

	let startTime = $state('');
	let endTime = $state('');

	type RunState = 'idle' | 'loading' | 'success' | 'error';
	let runState = $state<RunState>('idle');
	let runMessage = $state('');

	function toLocalInputValue(date: Date): string {
		const pad = (n: number) => String(n).padStart(2, '0');
		return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
	}

	async function loadFolders() {
		try {
			const res = await fetch(FOLDERS_URL);
			if (!res.ok) throw new Error(`HTTP ${res.status}`);
			const payload = (await res.json()) as { folders: string[] };
			folders = payload.folders;
			loadError = null;
		} catch (e) {
			loadError = e instanceof Error ? e.message : String(e);
		}
	}

	function toggleFolder(name: string, checked: boolean) {
		if (checked) selected.add(name);
		else selected.delete(name);
		loadPreview();
	}

	async function loadPreview() {
		if (selected.size === 0) {
			points = [];
			return;
		}
		previewState = 'loading';
		try {
			const dirs = [...selected].join(',');
			const res = await fetch(`${PREVIEW_URL}?dirs=${encodeURIComponent(dirs)}`);
			const payload = (await res.json()) as {
				timestamps?: string[];
				water_content?: number[];
				error?: string;
			};
			if (!res.ok) throw new Error(payload.error ?? `HTTP ${res.status}`);
			const timestamps = payload.timestamps ?? [];
			const waterContent = payload.water_content ?? [];
			points = timestamps.map((t, i) => ({ time: new Date(t), water_content: waterContent[i] }));
			if (points.length > 0) {
				startTime = toLocalInputValue(points[0].time);
				endTime = toLocalInputValue(points[points.length - 1].time);
			}
			previewState = 'idle';
		} catch (e) {
			previewState = 'error';
			previewMessage = e instanceof Error ? e.message : String(e);
		}
	}

	async function handleRun() {
		runState = 'loading';
		runMessage = '';
		try {
			const res = await fetch(RUN_URL, {
				method: 'POST',
				body: JSON.stringify({
					dirs: [...selected],
					start_time: startTime || undefined,
					end_time: endTime || undefined
				})
			});
			const payload = (await res.json()) as { output_dir?: string; error?: string };
			if (!res.ok) throw new Error(payload.error ?? `HTTP ${res.status}`);
			runState = 'success';
			runMessage = payload.output_dir ?? '';
		} catch (e) {
			runState = 'error';
			runMessage = e instanceof Error ? e.message : String(e);
		}
	}

	onMount(() => {
		loadFolders();
	});
</script>

<main class="mx-auto flex max-w-3xl flex-col gap-6 p-8">
	<header class="flex items-center justify-between">
		<h1 class="text-xl font-semibold">後処理 (Postprocess)</h1>
		<a href={resolve('/')} class="text-sm text-blue-600 underline">監視画面に戻る</a>
	</header>

	{#if loadError}
		<p class="border border-red-300 bg-red-50 p-4 text-red-700">
			フォルダ一覧の取得に失敗しました: {loadError}
		</p>
	{:else if folders.length === 0}
		<p class="text-gray-500">読み込み中...</p>
	{:else}
		<section class="flex flex-col gap-2 border p-3 text-sm">
			<h2 class="text-gray-500">処理対象フォルダ(複数選択可)</h2>
			<div class="flex max-h-48 flex-col gap-1 overflow-y-auto">
				{#each folders as name (name)}
					<label class="flex items-center gap-2">
						<input
							type="checkbox"
							checked={selected.has(name)}
							onchange={(e) => toggleFolder(name, e.currentTarget.checked)}
						/>
						{name}
					</label>
				{/each}
			</div>
		</section>

		{#if previewState === 'error'}
			<p class="border border-red-300 bg-red-50 p-2 text-sm text-red-700">
				概形の取得に失敗しました: {previewMessage}
			</p>
		{:else if selected.size > 0}
			<section class="flex flex-col gap-3 border p-3 text-sm">
				<h2 class="text-gray-500">概形 (Water Content)</h2>
				<div class="h-64">
					{#if previewState === 'loading'}
						<p class="text-gray-400">読み込み中...</p>
					{:else if points.length > 0}
						<LineChart
							data={points}
							x="time"
							y="water_content"
							padding={{ left: 56, bottom: 48 }}
							props={{
								xAxis: { label: 'Time', labelProps: { class: 'text-xs' } },
								yAxis: { label: 'Water Content', labelProps: { class: 'text-xs' } }
							}}
						/>
					{/if}
				</div>

				<div class="flex flex-wrap items-center gap-4">
					<div class="flex items-center gap-2">
						<label for="start-time" class="text-gray-500">開始</label>
						<input id="start-time" type="datetime-local" bind:value={startTime} class="border px-2 py-1" />
					</div>
					<div class="flex items-center gap-2">
						<label for="end-time" class="text-gray-500">終了</label>
						<input id="end-time" type="datetime-local" bind:value={endTime} class="border px-2 py-1" />
					</div>
				</div>

				<button
					class="w-fit rounded border px-3 py-1.5 disabled:opacity-50"
					onclick={handleRun}
					disabled={runState === 'loading'}
				>
					{runState === 'loading' ? '処理中...' : '実行'}
				</button>
			</section>
		{/if}

		{#if runState === 'success'}
			<p class="border border-green-300 bg-green-50 p-2 text-sm text-green-700">
				出力しました: {runMessage}
			</p>
		{:else if runState === 'error'}
			<p class="border border-red-300 bg-red-50 p-2 text-sm text-red-700">
				処理に失敗しました: {runMessage}
			</p>
		{/if}
	{/if}
</main>
