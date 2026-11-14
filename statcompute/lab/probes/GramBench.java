// Sandbox microbenchmark (indicative only): ways to accumulate a packed weighted Gram matrix
//   G = sum_i w_i x_i x_i'   over n rows with p covariates, row-major storage.
// V1: per-element Neumaier update with branches (the style of pprof_spark's NeumaierVector.addAt)
// V2: per-element branch-free TwoSum (same error term, so the same bits as V1)
// V3: plain rank-one updates into a buffer for chunks of 64 rows, each chunk merged by TwoSum
// V4: as V3, but indicator columns stored as index lists (only non-zero pairs are touched; uncentred)
import java.util.*;

public class GramBench {
    static final class Acc {
        final double[] sums, comps;
        Acc(int q) { sums = new double[q]; comps = new double[q]; }
        void addAt(int i, double x) {                      // V1: Neumaier, as in NeumaierVector.addAt
            double s = sums[i];
            double t = s + x;
            if (Double.isFinite(t)) {
                if (Math.abs(s) >= Math.abs(x)) comps[i] += (s - t) + x;
                else comps[i] += (x - t) + s;
            }
            sums[i] = t;
        }
        double valueAt(int i) { return Double.isFinite(sums[i]) ? sums[i] + comps[i] : sums[i]; }
        void reset() { Arrays.fill(sums, 0.0); Arrays.fill(comps, 0.0); }
    }

    static void v1(double[] x, double[] w, int n, int p, Acc a) {
        for (int r = 0; r < n; r++) {
            int off = r * p; int k = 0;
            for (int j = 0; j < p; j++) {
                double wx = w[r] * x[off + j];
                for (int l = j; l < p; l++) { a.addAt(k, wx * x[off + l]); k++; }
            }
        }
    }

    static void v2(double[] x, double[] w, int n, int p, Acc a) {
        final double[] sums = a.sums, comps = a.comps;
        for (int r = 0; r < n; r++) {
            int off = r * p; int base = 0;
            for (int j = 0; j < p; j++) {
                double wx = w[r] * x[off + j];
                int kb = base - j;
                for (int l = j; l < p; l++) {
                    double v = wx * x[off + l];
                    double s = sums[kb + l];
                    double t = s + v;
                    double z = t - s;
                    comps[kb + l] += (s - (t - z)) + (v - z);
                    sums[kb + l] = t;
                }
                base += p - j;
            }
        }
    }

    static void merge(double[] buf, Acc a) {
        final double[] sums = a.sums, comps = a.comps;
        for (int k = 0; k < buf.length; k++) {
            double v = buf[k];
            double s = sums[k];
            double t = s + v;
            double z = t - s;
            comps[k] += (s - (t - z)) + (v - z);
            sums[k] = t;
            buf[k] = 0.0;
        }
    }

    static void v3(double[] x, double[] w, int n, int p, Acc a, double[] buf, int chunk) {
        for (int r0 = 0; r0 < n; r0 += chunk) {
            int r1 = Math.min(n, r0 + chunk);
            for (int r = r0; r < r1; r++) {
                int off = r * p; int base = 0;
                for (int j = 0; j < p; j++) {
                    double wx = w[r] * x[off + j];
                    int kb = base - j;
                    for (int l = j; l < p; l++) buf[kb + l] += wx * x[off + l];
                    base += p - j;
                }
            }
            merge(buf, a);
        }
    }

    // V4: columns [0, pd) dense, columns [pd, p) indicators given as ascending index lists per row.
    static void v4(double[] xd, int pd, int[] idx, int[] start, double[] w, int n, int p, Acc a, double[] buf, int chunk, int[] rowBase) {
        for (int r0 = 0; r0 < n; r0 += chunk) {
            int r1 = Math.min(n, r0 + chunk);
            for (int r = r0; r < r1; r++) {
                int off = r * pd; double wr = w[r];
                int s0 = start[r], s1 = start[r + 1];
                for (int j = 0; j < pd; j++) {
                    double wx = wr * xd[off + j];
                    int kb = rowBase[j] - j;
                    for (int l = j; l < pd; l++) buf[kb + l] += wx * xd[off + l];
                    for (int t = s0; t < s1; t++) buf[kb + idx[t]] += wx;          // dense x indicator
                }
                for (int t = s0; t < s1; t++) {
                    int kb = rowBase[idx[t]] - idx[t];
                    for (int u = t; u < s1; u++) buf[kb + idx[u]] += wr;           // indicator x indicator
                }
            }
            merge(buf, a);
        }
    }

    static double time(Runnable f, int reps) {
        double best = Double.MAX_VALUE;
        for (int i = 0; i < reps; i++) { long t0 = System.nanoTime(); f.run(); best = Math.min(best, (System.nanoTime() - t0) * 1e-9); }
        return best;
    }

    public static void main(String[] args) {
        int n = Integer.parseInt(args[0]);
        double density = Double.parseDouble(args[1]);
        for (int p : new int[] {20, 50, 100}) {
            int q = p * (p + 1) / 2; int pd = p / 2;                 // half continuous, half indicators (as the scale kit)
            SplittableRandom rng = new SplittableRandom(79);
            double[] x = new double[n * p], xd = new double[n * pd], w = new double[n];
            int[] start = new int[n + 1]; int[] idx = new int[n * (p - pd)]; int nz = 0;
            for (int r = 0; r < n; r++) {
                w[r] = 0.05 + rng.nextDouble();
                for (int j = 0; j < pd; j++) { double v = Math.rint(rng.nextGaussian() * 64) / 64; x[r * p + j] = v; xd[r * pd + j] = v; }
                start[r] = nz;
                for (int j = pd; j < p; j++) if (rng.nextDouble() < density) { x[r * p + j] = 1.0; idx[nz++] = j; }
            }
            start[n] = nz;
            int[] rowBase = new int[p]; for (int j = 1; j < p; j++) rowBase[j] = rowBase[j - 1] + (p - (j - 1));
            Acc a1 = new Acc(q), a2 = new Acc(q), a3 = new Acc(q), a4 = new Acc(q); double[] buf = new double[q];
            final int P = p, PD = pd; final double[] X = x, XD = xd, W = w; final int[] IDX = idx, ST = start, RB = rowBase;
            for (int warm = 0; warm < 6; warm++) {                    // warm-up: let C2 compile every variant
                a1.reset(); v1(X, W, n, P, a1); a2.reset(); v2(X, W, n, P, a2);
                a3.reset(); v3(X, W, n, P, a3, buf, 64); a4.reset(); v4(XD, PD, IDX, ST, W, n, P, a4, buf, 64, RB);
            }
            double t1 = time(() -> { a1.reset(); v1(X, W, n, P, a1); }, 7);
            double t2 = time(() -> { a2.reset(); v2(X, W, n, P, a2); }, 7);
            double t3 = time(() -> { a3.reset(); v3(X, W, n, P, a3, buf, 64); }, 7);
            double t4 = time(() -> { a4.reset(); v4(XD, PD, IDX, ST, W, n, P, a4, buf, 64, RB); }, 7);
            boolean bits12 = true; double rel13 = 0, rel14 = 0, scale = 0;
            for (int k = 0; k < q; k++) scale = Math.max(scale, Math.abs(a1.valueAt(k)));
            for (int k = 0; k < q; k++) {
                if (Double.doubleToRawLongBits(a1.sums[k]) != Double.doubleToRawLongBits(a2.sums[k])
                 || Double.doubleToRawLongBits(a1.comps[k]) != Double.doubleToRawLongBits(a2.comps[k])) bits12 = false;
                rel13 = Math.max(rel13, Math.abs(a1.valueAt(k) - a3.valueAt(k)) / scale);
                rel14 = Math.max(rel14, Math.abs(a1.valueAt(k) - a4.valueAt(k)) / scale);
            }
            double elems = (double) n * q;
            System.out.printf(Locale.ROOT,
                "p=%3d  indicator density %.2f | V1 Neumaier %6.3fs (%.2f ns/elem) | V2 TwoSum %6.3fs x%.1f bit-identical=%b | V3 chunked %6.3fs x%.1f maxrel %.1e | V4 sparse-aware %6.3fs x%.1f maxrel %.1e%n",
                p, density, t1, t1 / elems * 1e9, t2, t1 / t2, bits12, t3, t1 / t3, rel13, t4, t1 / t4, rel14);
        }
    }
}
