<script lang="ts">
	import { onMount } from 'svelte';
	import { formatNanoseconds } from '$lib/format';
	import ImpulseResponseChart from '$lib/ImpulseResponseChart.svelte';
	import WaterContentTrendChart from '$lib/WaterContentTrendChart.svelte';

	const API_URL = 'http://127.0.0.1:5290/latest';
	const POLL_INTERVAL_MS = 1000;

	type LatestData = {
		frequencies: number[];
		s11_real: number[];
		s11_imag: number[];
		d_axis: number[];
		step_response: number[];
		impulse_response: number[];
		peak_distance: number;
		water_content: number;
	};

	let data = $state<LatestData | null>(null);
	let error = $state<string | null>(null);
	let lastUpdated = $state<Date | null>(null);

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
		<span
			class="flex items-center gap-2 text-sm"
			class:text-green-600={!error}
			class:text-red-600={!!error}
		>
			<span class="h-2 w-2 rounded-full" class:bg-green-600={!error} class:bg-red-600={!!error}
			></span>
			{error ? '切断' : '接続中'}
		</span>
	</header>

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
				<ImpulseResponseChart x={data.d_axis} y={data.impulse_response} />
			</div>
			<WaterContentTrendChart />
		</div>
		{#if lastUpdated}
			<p class="text-xs text-gray-400">最終更新: {lastUpdated.toLocaleTimeString()}</p>
		{/if}
	{/if}
</main>
