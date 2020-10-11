---
title: 
date: 2020-8-10
tags:
categories: ["Python笔记"]
mathjax: true
---

# np.delete

```python
def delete(arr, obj, axis=None):
    """
    Return a new array with sub-arrays along an axis deleted. For a one
    dimensional array, this returns those entries not returned by `arr[obj]`.

    Parameters
    ----------
    arr : array_like
      Input array.
    obj : slice, int or array of ints
      Indicate which sub-arrays to remove.
    axis : int, optional
      The axis along which to delete the subarray defined by `obj`.
      If `axis` is None, `obj` is applied to the flattened array.

    Returns
    -------
    out : ndarray
        A copy of `arr` with the elements specified by `obj` removed. Note
        that `delete` does not occur in-place. If `axis` is None, `out` is
        a flattened array.

    See Also
    --------
    insert : Insert elements into an array.
    append : Append elements at the end of an array.

    Notes
    -----
    Often it is preferable to use a boolean mask. For example:
    >>> mask = np.ones(len(arr), dtype=bool)
    >>> mask[[0,2,4]] = False
    >>> result = arr[mask,...]
    Is equivalent to `np.delete(arr, [0,2,4], axis=0)`, but allows further
    use of `mask`.

    Examples
    --------
    >>> arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    >>> arr
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])
    >>> np.delete(arr, 1, 0)
    array([[ 1,  2,  3,  4],
           [ 9, 10, 11, 12]])

    >>> np.delete(arr, np.s_[::2], 1)
    array([[ 2,  4],
           [ 6,  8],
           [10, 12]])
    >>> np.delete(arr, [1,3,5], None)
    array([ 1,  3,  5,  7,  8,  9, 10, 11, 12])

    """
    wrap = None
    if type(arr) is not ndarray:
        try:
            wrap = arr.__array_wrap__
        except AttributeError:
            pass

    arr = asarray(arr)
    ndim = arr.ndim
    if axis is None:
        if ndim != 1:
            arr = arr.ravel()
        ndim = arr.ndim;
        axis = ndim-1;
    if ndim == 0:
        warnings.warn("in the future the special handling of scalars "
                      "will be removed from delete and raise an error",
                      DeprecationWarning)
        if wrap:
            return wrap(arr)
        else:
            return arr.copy()

    slobj = [slice(None)]*ndim
    N = arr.shape[axis]
    newshape = list(arr.shape)

    if isinstance(obj, slice):
        start, stop, step = obj.indices(N)
        xr = range(start, stop, step)
        numtodel = len(xr)

        if numtodel <= 0:
            if wrap:
                return wrap(arr.copy())
            else:
                return arr.copy()

        # Invert if step is negative:
        if step < 0:
            step = -step
            start = xr[-1]
            stop = xr[0] + 1

        newshape[axis] -= numtodel
        new = empty(newshape, arr.dtype, arr.flags.fnc)
        # copy initial chunk
        if start == 0:
            pass
        else:
            slobj[axis] = slice(None, start)
            new[slobj] = arr[slobj]
        # copy end chunck
        if stop == N:
            pass
        else:
            slobj[axis] = slice(stop-numtodel, None)
            slobj2 = [slice(None)]*ndim
            slobj2[axis] = slice(stop, None)
            new[slobj] = arr[slobj2]
        # copy middle pieces
        if step == 1:
            pass
        else:  # use array indexing.
            keep = ones(stop-start, dtype=bool)
            keep[:stop-start:step] = False
            slobj[axis] = slice(start, stop-numtodel)
            slobj2 = [slice(None)]*ndim
            slobj2[axis] = slice(start, stop)
            arr = arr[slobj2]
            slobj2[axis] = keep
            new[slobj] = arr[slobj2]
        if wrap:
            return wrap(new)
        else:
            return new

    _obj = obj
    obj = np.asarray(obj)
    # After removing the special handling of booleans and out of
    # bounds values, the conversion to the array can be removed.
    if obj.dtype == bool:
        warnings.warn("in the future insert will treat boolean arrays "
                      "and array-likes as boolean index instead "
                      "of casting it to integer", FutureWarning)
        obj = obj.astype(intp)
    if isinstance(_obj, (int, long, integer)):
        # optimization for a single value
        obj = obj.item()
        if (obj < -N or obj >=N):
            raise IndexError("index %i is out of bounds for axis "
                             "%i with size %i" % (obj, axis, N))
        if (obj < 0): obj += N
        newshape[axis]-=1;
        new = empty(newshape, arr.dtype, arr.flags.fnc)
        slobj[axis] = slice(None, obj)
        new[slobj] = arr[slobj]
        slobj[axis] = slice(obj, None)
        slobj2 = [slice(None)]*ndim
        slobj2[axis] = slice(obj+1, None)
        new[slobj] = arr[slobj2]
    else:
        if obj.size == 0 and not isinstance(_obj, np.ndarray):
            obj = obj.astype(intp)
        if not np.can_cast(obj, intp, 'same_kind'):
            # obj.size = 1 special case always failed and would just
            # give superfluous warnings.
            warnings.warn("using a non-integer array as obj in delete "
                "will result in an error in the future", DeprecationWarning)
            obj = obj.astype(intp)
        keep = ones(N, dtype=bool)

        # Test if there are out of bound indices, this is deprecated
        inside_bounds = (obj < N) & (obj >= -N)
        if not inside_bounds.all():
            warnings.warn("in the future out of bounds indices will raise an "
                          "error instead of being ignored by `numpy.delete`.",
                          DeprecationWarning)
            obj = obj[inside_bounds]
        positive_indices = obj >= 0
        if not positive_indices.all():
            warnings.warn("in the future negative indices will not be ignored "
                          "by `numpy.delete`.", FutureWarning)
            obj = obj[positive_indices]

        keep[obj,] = False
        slobj[axis] = keep
        new = arr[slobj]

    if wrap:
        return wrap(new)
    else:
        return new
```

**参数说明：**

**arr**: 待处理array \
**obj**: slice, int or array of ints Indicate which sub-arrays to remove. \
**axis**，int, optional \
      The axis along which to delete the subarray defined by `obj`. \
      If `axis` is None, `obj` is applied to the flattened array.



**代码示例：**
```python
if True:
	arr = np.array([(1, 2, 3, 4), (5, 6, 7, 8), (11, 12, 13, 14), (15, 16, 17, 18)])
	print(arr)
    # output:
    """
[[ 1  2  3  4]
 [ 5  6  7  8]
 [11 12 13 14]
 [15 16 17 18]]
    """

	rslt = np.delete(arr, [0, 2], axis=0)
	print(rslt)
    # output:
    """
[[ 5  6  7  8]
 [15 16 17 18]]
    """

	rslt = np.delete(arr, [0, 2], axis=1)
	print(rslt)
    # output:
    """
[[ 2  4]
 [ 6  8]
 [12 14]
 [16 18]]
    """

	rslt = np.delete(arr, 1, axis=0)
	print(rslt)
    # output:
    """
[[ 1  2  3  4]
 [11 12 13 14]
 [15 16 17 18]]
    """

	rslt = np.delete(arr, 1)
	print(rslt)
    # output:
    """
[ 1  3  4  5  6  7  8 11 12 13 14 15 16 17 18]
    """
```


