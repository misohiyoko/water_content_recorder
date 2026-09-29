<script lang="ts">
	import { onMount } from 'svelte';
	import { resolve } from '$app/paths';
	import { formatNanoseconds } from '$lib/format';
	import ImpulseResponseChart from '$lib/ImpulseResponseChart.svelte';
	import WaterContentTrendChart from '$lib/WaterContentTrendChart.svelte';

	const API_URL = 'http://127.0.0.1:5290/latest';
	const EXPORT_URL = 'http://127.0.0.1:5290/export';
	const CALIBRATE_URL = 'http://127.0.0.1:5290/calibrate';
	const POLL_INTERVAL_MS = 1000;

	type LatestData = {
		frequencies: number[];
		s11_real: number[];
		s11_imag: number[];
		t_axis: number[];
		step_response: number[];
		impulse_response: number[];
		peak_positions: number[];
		peak_distance: number;
		water_content: number;
	};

	let data = $state<LatestData | null>(null);
	let error = $state<string | null>(null);
	let lastUpdated = $state<Date | null>(null);
	let maxPeaks = $state(3);

	type ExportState = 'idle' | 'loading' | 'success' | 'error';
	let exportState = $state<ExportState>('idle');
	let exportMessage = $state('');

	async function handleExport() {
		exportState = 'loading';
		exportMessage = '';
		try {
			const res = await fetch(EXPORT_URL, { method: 'POST' });
			const payload = (await res.json()) as { output_dir?: string; error?: string };
			if (!res.ok) throw new Error(payload.error ?? `HTTP ${res.status}`);
			exportState = 'success';
			exportMessage = payload.output_dir ?? '';
		} catch (e) {
			exportState = 'error';
			exportMessage = e instanceof Error ? e.message : String(e);
		}
	}

	type CalibrateState = 'idle' | 'loading' | 'success' | 'error';
	let referenceWaterContent = $state<number | undefined>(undefined);
	let calibrateState = $state<CalibrateState>('idle');
	let calibrateMessage = $state('');

	async function handleCalibrate() {
		if (referenceWaterContent === undefined || Number.isNaN(referenceWaterContent)) return;
		calibrateState = 'loading';
		calibrateMessage = '';
		try {
			const res = await fetch(CALIBRATE_URL, {
				method: 'POST',
				body: JSON.stringify({ reference_water_content: referenceWaterContent })
			});
			const payload = (await res.json()) as { calibration_coefficient?: number; error?: string };
			if (!res.ok) throw new Error(payload.error ?? `HTTP ${res.status}`);
			calibrateState = 'success';
			calibrateMessage = `${payload.calibration_coefficient}`;
		} catch (e) {
			calibrateState = 'error';
			calibrateMessage = e instanceof Error ? e.message : String(e);
		}
	}

	onMount(() => {
		let cancelled = false;

		const poll = async () => {
			try {
				const res = await fetch(API_URL);
				if (!res.ok) throw new Error(`HTTP ${res.status}`);
				const payload = (await res.json()) as LatestData;
				if (cancelled) return;
				data = payload;
				error = null;
				lastUpdated = new Date();
			} catch (e) {
				if (cancelled) return;
				error = e instanceof Error ? e.message : String(e);
			}
		};

		poll();
		const id = setInterval(poll, POLL_INTERVAL_MS);
		return () => {
			cancelled = true;
			clearInterval(id);
		};
	});
</script>

<main class="mx-auto flex max-w-6xl flex-col gap-6 p-8">
	<header class="flex items-center justify-between">
		<h1 class="text-xl font-semibold">Water Content Monitor</h1>
		<div class="flex items-center gap-4">
			<a href={resolve('/postprocess')} class="text-sm text-blue-600 underline">後処理</a>
			<span
				class="flex items-center gap-2 text-sm"
				class:text-green-600={!error}
				class:text-red-600={!!error}
			>
				<span class="h-2 w-2 rounded-full" class:bg-green-600={!error} class:bg-red-600={!!error}
				></span>
				{error ? '切断' : '接続中'}
			</span>
		</div>
	</header>

	<div class="flex flex-wrap items-center gap-4 border p-3 text-sm">
		<button
			class="rounded border px-3 py-1.5 disabled:opacity-50"
			onclick={handleExport}
			disabled={exportState === 'loading'}
		>
			{exportState === 'loading' ? '出力中...' : 'データ出力'}
		</button>

		<div class="flex items-center gap-2">
			<label for="reference-water-content" class="text-gray-500">実測水分量(%)</label>
			<input
				id="reference-water-content"
				type="number"
				step="0.01"
				bind:value={referenceWaterContent}
				class="w-24 border px-2 py-1"
			/>
			<button
				class="rounded border px-3 py-1.5 disabled:opacity-50"
				onclick={handleCalibrate}
				disabled={calibrateState === 'loading' ||
					referenceWaterContent === undefined ||
					Number.isNaN(referenceWaterContent)}
			>
				{calibrateState === 'loading' ? 'キャリブレーション中...' : 'キャリブレーション'}
			</button>
		</div>
	</div>

	{#if exportState === 'success'}
		<p class="border border-green-300 bg-green-50 p-2 text-sm text-green-700">
			出力しました: {exportMessage}
		</p>
	{:else if exportState === 'error'}
		<p class="border border-red-300 bg-red-50 p-2 text-sm text-red-700">
			出力に失敗しました: {exportMessage}
		</p>
	{/if}

	{#if calibrateState === 'success'}
		<p class="border border-green-300 bg-green-50 p-2 text-sm text-green-700">
			校正係数を更新しました: {calibrateMessage}
		</p>
	{:else if calibrateState === 'error'}
		<p class="border border-red-300 bg-red-50 p-2 text-sm text-red-700">
			キャリブレーションに失敗しました: {calibrateMessage}
		</p>
	{/if}

	{#if error}
		<p class="border border-red-300 bg-red-50 p-4 text-red-700">
			API ({API_URL}) に接続できません: {error}
		</p>
	{:else if !data}
		<p class="text-gray-500">読み込み中...</p>
	{:else}
		<div class="grid grid-cols-2 items-stretch gap-6">
			<div class="flex flex-col gap-6">
				<div class="grid grid-cols-2 gap-4">
					<div class="border p-4">
						<div class="text-sm text-gray-500">Water Content</div>
						<div class="text-3xl font-bold">{data.water_content.toFixed(3)}</div>
					</div>
					<div class="border p-4">
						<div class="text-sm text-gray-500">Peak Distance</div>
						<div class="text-3xl font-bold">{formatNanoseconds(data.peak_distance)}</div>
					</div>
				</div>
				<div class="flex items-center justify-end gap-2 text-sm text-gray-500">
					<label for="max-peaks">表示ピーク数</label>
					<input
						id="max-peaks"
						type="number"
						min="0"
						max={data.peak_positions.length}
						bind:value={maxPeaks}
						class="w-16 border px-2 py-1"
					/>
				</div>
				<ImpulseResponseChart
					x={data.t_axis.map((t) => t * 1e9)}
					y={data.impulse_response}
					peakPositions={data.peak_positions.map((t) => t * 1e9)}
					{maxPeaks}
				/>
			</div>
			<WaterContentTrendChart />
		</div>
		{#if lastUpdated}
			<p class="text-xs text-gray-400">最終更新: {lastUpdated.toLocaleTimeString()}</p>
		{/if}
	{/if}
</main>
