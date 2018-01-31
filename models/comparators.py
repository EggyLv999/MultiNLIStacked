from __future__ import print_function, division

import  sys

import dynet as dy

def comparator_mou(hx, hy):
    delta = dy.concatenate([hx, hy, dy.cmult(hx, hy), dy.abs(hx - hy)])
    return delta

def comparator_outer(hx, hy):
    outer_product = hx * dy.transpose(hy)
    print(outer_product.dim(), file=sys.stderr)
    outer_product = dy.reshape(outer_product, ((hx.dim()[0][0]) ** 2,))
    fused_product = dy.concatenate([outer_product, hx, hy])
    return fused_product

def comparator_image(hx, hy):
    (d,), bsize = hx.dim()
    ones = dy.ones((d, 1), bsize)
    hx = dy.concatenate_cols([hx, ones])
    hx = dy.transpose(dy.concatenate_cols([hy, ones]))
    outer_product = hx * dy.transpose(hy)
    outer_product = dy.reshape(outer_product, ((hx.dim()[0][0]) ** 2,))
    fused_product = dy.concatenate([outer_product, hx, hy])
    return fused_product

def comparator_none(hx, hy):
    return hx, hy

def comparator_concat(hx, hy):
    return dy.concatenate([hx, hy])

comparators = {
    'mou': comparator_mou,
    'outer': comparator_outer,
    'none': comparator_none,
    'concat': comparator_concat,
}
