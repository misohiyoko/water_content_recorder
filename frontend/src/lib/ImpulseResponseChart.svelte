<script lang="ts">
	import { LineChart } from 'layerchart';

	let { x, y }: { x: number[]; y: number[] } = $props();

	const data = $derived(x.map((xi, i) => ({ x: xi, y: y[i] })));
</script>

<section class="chart-root border p-4">
	<h2 class="mb-2 text-sm text-gray-500">Impulse Response</h2>
	<div class="h-64 w-full">
		<LineChart
			{data}
			x="x"
			y="y"
			props={{
				xAxis: { label: 'Distance (m)' },
				yAxis: { label: 'Amplitude' }
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
