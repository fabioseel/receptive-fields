# Prove SPD Convolutions are just strided with large kernels

## Preliminaries

Input:
$f: (c, w, h)$

Convolutional Kernel:
$g: (c, k, k)$

Application of Convolution:
$$(f*g)[m,n] = \sum_i^k\sum_j^k\sum_l^c g[l,i,j]f[l,m-i,n-j]$$

Strided Convolution with stride $s$:
$$(f*g)[m,n] = \sum_i^k\sum_j^k\sum_l^c g[l,i,j]f[l,s * m-i, s*n-j]$$

Space to Depth with downsampling factor $s$:
$$ (c, w, h) \to  \left(cs^2, \frac{w}{s}, \frac{h}{s} \right)$$
or inverse:
$$ (c_s, w_s, h_s) \to \left(\frac{c_s}{s^2},sw_s, sh_s \right) $$

which means that:


## 1d Prove:

Input: $f(c, w)$

Kernel: $g(c,k)$

Convolution:
$$(f*g)[m] = \sum_i^k\sum_l^c g[l,i]f[l,m-i]$$

Strided Convolution:
$$(f*g)[m] = \sum_i^k\sum_l^c g[l,i]f[l,sm-i]$$

Space to Depth with scale $s$:
$$f(l,i) \to f_s\left(sl + i \mod s, \left\lfloor\frac{i}{s}\right\rfloor\right)$$
and the inverse:
$$f_s(l,i) \to f\left(\left\lfloor\frac{l}{s}\right\rfloor, si+l\mod s\right)$$
$$f_s(l,i) \to f\left(\left\lfloor\frac{l}{s}\right\rfloor, si+l-\left\lfloor\frac{l}{s}\right\rfloor\right)$$
importantly, this can also be applied to the kernel!

So lets apply a convolution to g_s and then rewrite it in terms of g:

$$(f*g)[m] = \sum_i^k\sum_k^c g_s[l,i]f[l,m-i] \\
= \sum_i^k\sum_l^c g_s\left[\left\lfloor\frac{l}{s}\right\rfloor, si+l\right]f[l,m-i] \\$$

Now lets try to rewrite this with 
$$= \sum_i^k\sum_l^c g_s\left[\left\lfloor\frac{l}{s}\right\rfloor, si+l\mod s\right]f\left[\left\lfloor\frac{l}{s}\right\rfloor, s(m-i)+l\mod s\right] \\
= $$