# Computational classification of rank-2 diagonal reflections

The companion program [`prove_rank2_reflections.py`](prove_rank2_reflections.py)
is a dependency-free exhaustive verification of the classification.

## Reduction to a finite search

Write

\[
P(n)=c_0+\sum_i a_i n_i+\sum_{i<j}b_{ij}n_i n_j,
\qquad n_i\in\{0,1\}.
\]

The number operators commute and satisfy \(n_i^2=n_i\), so their common
eigenbasis is labeled by the Boolean cube. Consequently, \(P^2=1\) if and
only if \(P(n)\in\{-1,1\}\) at every vertex of that cube.

At the all-zero vertex, \(c_0=P(0)=\pm1\). Multiplication by the allowed global
phase fixes \(c_0=1\). Values at vertices of Hamming weight one and two then
give

\[
a_i=P(e_i)-1,
\qquad
b_{ij}=P(e_i+e_j)-1-a_i-a_j.
\]

Thus, the values of \(P\) at those vertices uniquely determine all its
coefficients. Enumerating their signs is therefore exhaustive, not a search
over an assumed coefficient box. For \(m\) indices it tests exactly

\[
2^{m+\binom m2}
\]

candidates and checks each candidate on all \(2^m\) occupation vectors.

The same equations immediately imply

\[
a_i\in\{-2,0\},\qquad b_{ij}\in\{-2,0,2,4\}.
\]

The program calls an index *essential* when it occurs in at least one nonzero
linear or quadratic monomial. It discards nonessential dummy indices and
canonically minimizes each coefficient tuple over all index permutations. Its
output is

```text
m  candidates  solutions  essential  essential-orbits
1           2          2          1                 1
2           8          8          5                 4
3          64         35         16                 4
4        1024        111         12                 1
5       32768        276          0                 0
```

The canonical orbit sets are asserted to equal the transcription of
\(p_1,\ldots,p_{10}\), not merely counted. The orbit counts are
\(1+4+4+1=10\).

Run the certificate with

```bash
python prove_rank2_reflections.py
```

Any omitted or additional orbit, a failed reflection identity, or a mismatch
with a displayed representative makes the program terminate with an assertion
failure.

## Why more than four essential indices are impossible

An exhaustive search at \(m=5\), by itself, does not logically rule out a
solution first appearing at \(m=6\). The following standard short argument
closes that gap.

Change variables from \(n_i\in\{0,1\}\) to \(x_i\in\{-1,1\}\). This preserves
degree, so a reflection becomes a Boolean-valued Fourier polynomial

\[
f(x)=\sum_{|S|\leq2}\widehat f(S)\prod_{i\in S}x_i.
\]

For a relevant variable \(i\), the discrete derivative obtained by flipping
\(x_i\) is a nonzero polynomial of degree at most one in the remaining
variables. A nonzero affine polynomial on a Boolean cube is nonzero on at
least half of its vertices: choose a variable having a nonzero coefficient
and pair vertices that differ only in that variable; the polynomial cannot
vanish at both endpoints. Because \(f\) is \(\pm1\)-valued, this says that the
influence of every relevant variable obeys

\[
\operatorname{Inf}_i(f)\geq\tfrac12.
\]

On the other hand, Parseval's identity gives

\[
\sum_i\operatorname{Inf}_i(f)
=\sum_S |S|\widehat f(S)^2
\leq2\sum_S\widehat f(S)^2=2.
\]

If \(r\) variables are relevant, then \(r/2\leq2\), hence \(r\leq4\). This
proves for arbitrarily many named spin orbitals that a quadratic reflection
can depend essentially on no more than four of them. Together with the exact
search through four essential indices, it proves the stated classification.
