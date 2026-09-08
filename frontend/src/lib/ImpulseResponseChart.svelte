<script lang="ts">
	import { LineChart } from 'layerchart';

	let {
		x,
		y,
		peakPositions = [],
		maxPeaks = 3
	}: { x: number[]; y: number[]; peakPositions?: number[]; maxPeaks?: number } = $props();

	const data = $derived(x.map((xi, i) => ({ x: xi, y: y[i] })));

	// peakPositionsは高さ降順なので、先頭からmaxPeaks件が表示対象の順位になる
	const visiblePeaks = $derived(peakPositions.slice(0, maxPeaks));

	function nearestY(px: number): number {
		let bestIdx = 0;
		let bestDiff = Infinity;
		for (let i = 0; i < x.length; i++) {
			const diff = Math.abs(x[i] - px);
			if (diff < bestDiff) {
				bestDiff = diff;
				bestIdx = i;
			}
		}
		return y[bestIdx] ?? 0;
	}

	const peakPoints = $derived(
		visiblePeaks.map((px, i) => ({ x: px, y: nearestY(px), rank: i + 1 }))
	);
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
				xAxis: { label: 'Time (ns)', labelProps: { class: 'text-xs' } },
				yAxis: { label: 'Amplitude', labelProps: { class: 'text-xs' } }
			}}
		>
			{#snippet aboveMarks({ context })}
				{#each peakPoints as peak (peak.rank)}
					<circle
						cx={context.xScale(peak.x)}
						cy={context.yScale(peak.y)}
						r="4"
						class="peak-marker"
					/>
					<text
						x={context.xScale(peak.x)}
						y={context.yScale(peak.y) - 8}
						class="peak-label"
						text-anchor="middle"
					>
						{peak.rank}
					</text>
				{/each}
			{/snippet}
		</LineChart>
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
	.peak-marker {
		fill: #d6432a;
		stroke: white;
		stroke-width: 1.5;
	}
	.peak-label {
		font-size: 11px;
		fill: #d6432a;
		font-weight: 600;
	}
	@media (prefers-color-scheme: dark) {
		.peak-marker {
			fill: #e56b39;
		}
		.peak-label {
			fill: #e56b39;
		}
	}
</style>
