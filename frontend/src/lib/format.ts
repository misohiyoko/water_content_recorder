export function formatNanoseconds(seconds: number): string {
	return `${(seconds * 1e9).toFixed(3)} ns`;
}
