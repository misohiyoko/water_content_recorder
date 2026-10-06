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

	let points = $state<{ time: Date; water_content: number | null }[]>([]);
	let previewState = $state<'idle' | 'loading' | 'error'>('idle');
	let previewMessage = $state('');

	let startTime = $state('');
	let endTime = $state('');

	// サーバーとの時刻のやり取りはUNIX時刻(ms)で行い、日時の入力・表示だけをブラウザのローカル時刻(JST)で行う
	const toMs = (v: string) => (v ? new Date(v).getTime() : undefined);
	// 古いリクエストの応答が後から届いて表示を上書きしないよう、最新のリクエスト番号だけを採用する
	let previewSeq = 0;
	let rangeTimer: ReturnType<typeof setTimeout> | undefined;

	type RunState = 'idle' | 'loading' | 'success' | 'error';
	let runState = $state<RunState>('idle');
	let runMessage = $state('');
	let generateCombinedParquet = $state(true);
	let generateImpulseResponse = $state(true);

	function toLocalInputValue(date: Date): string {
		const pad = (n: number) => String(n).padStart(2, '0');
		return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
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
		loadPreview(true);
	}

	// 開始/終了時刻が変わったら、その範囲だけを再取得する(全体の間引き済みデータを絞ると点が粗くなるため)
	function onRangeChange() {
		clearTimeout(rangeTimer);
		rangeTimer = setTimeout(() => loadPreview(false), 300);
	}

	// resetRange: フォルダ選択が変わったときは範囲を全体に戻す。範囲変更時は入力値で絞り込んで取得する
	async function loadPreview(resetRange: boolean) {
		const seq = ++previewSeq;
		if (selected.size === 0) {
			points = [];
			return;
		}
		previewState = 'loading';
		try {
			const params = new URLSearchParams({ dirs: [...selected].join(',') });
			const startMs = resetRange ? undefined : toMs(startTime);
			const endMs = resetRange ? undefined : toMs(endTime);
			if (startMs !== undefined) params.set('start_ms', String(startMs));
			if (endMs !== undefined) params.set('end_ms', String(endMs));
			const res = await fetch(`${PREVIEW_URL}?${params}`);
			const payload = (await res.json()) as {
				times_ms?: number[];
				water_content?: (number | null)[];
				error?: string;
			};
			if (seq !== previewSeq) return;
			if (!res.ok) throw new Error(payload.error ?? `HTTP ${res.status}`);
			const timesMs = payload.times_ms ?? [];
			const waterContent = payload.water_content ?? [];
			points = timesMs.map((t, i) => ({ time: new Date(t), water_content: waterContent[i] }));
			if (resetRange && points.length > 0) {
				startTime = toLocalInputValue(points[0].time);
				// 入力は秒単位なので、最後のレコードが範囲から漏れないよう終了は秒の切り上げにする
				endTime = toLocalInputValue(new Date(Math.ceil(timesMs[timesMs.length - 1] / 1000) * 1000));
			}
			previewState = 'idle';
		} catch (e) {
			if (seq !== previewSeq) return;
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
					start_ms: toMs(startTime),
					end_ms: toMs(endTime),
					generate_combined_parquet: generateCombinedParquet,
					generate_impulse_response: generateImpulseResponse
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
								xAxis: { label: 'Time (JST)', labelProps: { class: 'text-xs' } },
								yAxis: { label: 'Water Content', labelProps: { class: 'text-xs' } }
							}}
						/>
					{/if}
				</div>

				<div class="flex flex-wrap items-center gap-4">
					<div class="flex items-center gap-2">
						<label for="start-time" class="text-gray-500">開始</label>
						<input
							id="start-time"
							type="datetime-local"
							step="1"
							bind:value={startTime}
							oninput={onRangeChange}
							class="border px-2 py-1"
						/>
					</div>
					<div class="flex items-center gap-2">
						<label for="end-time" class="text-gray-500">終了</label>
						<input
							id="end-time"
							type="datetime-local"
							step="1"
							bind:value={endTime}
							oninput={onRangeChange}
							class="border px-2 py-1"
						/>
					</div>
				</div>

				<div class="flex flex-wrap items-center gap-4 text-gray-500">
					<label class="flex items-center gap-2">
						<input type="checkbox" bind:checked={generateCombinedParquet} />
						combined.parquetを出力する
					</label>
					<label class="flex items-center gap-2">
						<input type="checkbox" bind:checked={generateImpulseResponse} />
						impulse_response.htmlを出力する
					</label>
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
