## split | The 3-node network and Set-Splitting (Theorems 2–3)
**3-node network:** $n$ binary inputs, **two** hidden linear threshold units $N_1,N_2$, one threshold output $N_3$. Training problem: given labelled examples, do weights exist that classify all of them correctly? Theorem 2: NP-complete.
Geometric form (Sec. 4.3.1): either one plane separates the classes, or two planes cut space so that one quadrant contains all and only the “+” points.
### Reduction from Set-Splitting (Theorem 3, AND output)
Elements $s_1,\dots,s_n$, subsets $c_1,\dots,c_k$. Examples: origin “+”, unit vectors $p_i$ “−”, and $\sum_{j\in c_i}p_j$ “+”. Given a split $(S_1,S_2)$:
$$a_j=\begin{cases}-1&s_j\in S_1\\ n&\text{otherwise}\end{cases},\qquad \text{plane } a\cdot x=-\tfrac12\quad(\text{same for } b \text{ with } S_2)$$
and the net outputs “+” iff $a\cdot x>-\tfrac12$ and $b\cdot x>-\tfrac12$. Every subset is cut by the split $\iff$ all examples are classified correctly.
### Paper's example on screen
$S=\{s_1,s_2,s_3\}$, $C=\{\{s_1,s_2\},\{s_2,s_3\}\}$: “−” at 100, 010, 001; “+” at 000, 110, 011. Only $S_1=\{s_2\}$ (or its swap) works.
> Check script: all 8 assignments tested; the six points are not separable by one plane (so a single threshold unit cannot fit them).
> The survey says “three hidden nodes”; the paper's network has two hidden nodes plus one output node.

## xor | Representation matters (Sec. 5)
Also NP-complete: equal hidden weights, $k$ hidden units with an AND output, XOR output with ternary features. But the XOR-output 3-node net's decision rule is linear in the features $(x_i)\cup(x_ix_j)$:
$$x_1\oplus x_2=x_1+x_2-2x_1x_2$$
so a perceptron over $O(n^2)$ inputs, trained by linear programming in polynomial time, is at least as expressive.
