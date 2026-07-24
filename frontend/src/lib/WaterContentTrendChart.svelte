<script lang="ts">
	import { onMount } from 'svelte';
	import { LineChart } from 'layerchart';

	const API_URL = 'http://127.0.0.1:8000/history';
	const POLL_INTERVAL_MS = 10_000;

	type HistoryData = {
		timestamps: string[];
		water_content: number[];
	};

	let points = $state<{ time: Date; water_content: number }[]>([]);

	onMount(() => {
		let cancelled = false;

		const poll = async () => {
			try {
				const res = await fetch(API_URL);
				if (!res.ok) return;
				const payload = (await res.json()) as HistoryData;
				if (cancelled) return;
				points = payload.timestamps.map((t, i) => ({
					time: new Date(t),
					water_content: payload.water_content[i]
				}));
			} catch {
				// keep showing the last known trend on a transient poll failure
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

<section class="chart-root flex h-full flex-col border p-4">
	<h2 class="mb-2 text-sm text-gray-500">Water Content (last 1h)</h2>
	<div class="min-h-0 flex-1">
		{#if points.length > 0}
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
		{:else}
			<p class="text-gray-400">読み込み中...</p>
		{/if}
	</div>
</section>

<style>
	.chart-root {
		--color-primary: #eb6834;
	}
	@media (prefers-color-scheme: dark) {
		.chart-root {
			--color-primary: #d95926;
		}
	}
</style>
