<script lang="ts">
	import { LineChart } from 'layerchart';

	let { x, y }: { x: number[]; y: number[] } = $props();

	const data = $derived(x.map((xi, i) => ({ x: xi, y: y[i] })));
</script>

<section class="chart-root border p-4">
	<h2 class="mb-2 text-sm text-gray-500">Impulse Response</h2>
	<div class="h-72 w-full">
		<LineChart
			{data}
			x="x"
			y="y"
			padding={{ left: 56, bottom: 48 }}
			props={{
				xAxis: { label: 'Distance (m)', labelProps: { class: 'text-xs' } },
				yAxis: { label: 'Amplitude', labelProps: { class: 'text-xs' } }
			}}
		/>
	</div>
</section>

<style>
	.chart-root {
		--color-primary: #2a78d6;
	}
	@media (prefers-color-scheme: dark) {
		.chart-root {
			--color-primary: #3987e5;
		}
	}
</style>
