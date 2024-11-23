# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


Timing summary from script posted in Ed Discusion, on colab
Size: 64
    fast: 0.00315
    gpu: 0.00763
Size: 128
    fast: 0.01561
    gpu: 0.01495
Size: 256
    fast: 0.09782
    gpu: 0.05213
Size: 512
    fast: 1.21924
    gpu: 0.28147
Size: 1024
    fast: 8.00809
    gpu: 0.88502

Dataset: Split, hidden 100, lr: 0.05
CPU, stopped after correct >= npts -1, time in seconds, had to stop early for gpu limit
Epoch  0  loss  5.437543860563423 correct 34
Time taken for 10 epochs:  1.1298012733459473
Epoch  10  loss  4.396465620777267 correct 34
Time taken for 10 epochs:  1.1330957412719727
Epoch  20  loss  5.4698390202196885 correct 41
Time taken for 10 epochs:  1.1127901077270508
Epoch  30  loss  4.280452030217447 correct 38
Time taken for 10 epochs:  1.1325325965881348
Epoch  40  loss  4.5068013114152485 correct 47
Time taken for 10 epochs:  2.2830824851989746
Epoch  50  loss  1.4908953234246018 correct 47
Time taken for 10 epochs:  2.4320409297943115
Epoch  60  loss  1.6225188258196752 correct 47
Time taken for 10 epochs:  1.1222341060638428
Epoch  70  loss  3.2288473453647537 correct 46
Time taken for 10 epochs:  1.1392590999603271
Epoch  80  loss  2.1836166328962294 correct 50

GPU, stopped after correct >= npts -1, time in seconds, had to stop early for gpu limit
Epoch  0  loss  6.707355985784231 correct 29
Time taken for 10 epochs:  18.069041967391968
Epoch  10  loss  5.00763617599726 correct 39
Time taken for 10 epochs:  18.011926889419556
Epoch  20  loss  5.276593720360865 correct 41
Time taken for 10 epochs:  17.151755571365356
Epoch  30  loss  4.48320616353456 correct 39
Time taken for 10 epochs:  17.610060214996338
Epoch  40  loss  8.517905833922057 correct 40
Time taken for 10 epochs:  17.591301202774048
Epoch  50  loss  3.229192316121171 correct 46
Time taken for 10 epochs:  17.24885320663452
Epoch  60  loss  2.413598219256898 correct 42
Time taken for 10 epochs:  17.876571655273438
Epoch  70  loss  3.2152579297442623 correct 48
Time taken for 10 epochs:  17.186405420303345
Epoch  80  loss  2.991568456588007 correct 49
Time taken for 10 epochs:  14.875287532806396

Dataset: simple, hidden 100, lr: 0.05
CPU, stopped after correct >= npts -1, time in seconds, had to stop early for gpu limit
Epoch  0  loss  10.398920090353725 correct 23
Time taken for 10 epochs:  17.366689920425415
Epoch  10  loss  6.540796253321593 correct 37
Time taken for 10 epochs:  17.681689977645874
Epoch  20  loss  4.317599832086218 correct 46
Time taken for 10 epochs:  18.206380128860474
Epoch  30  loss  3.649445142556525 correct 48
Time taken for 10 epochs:  17.172802209854126
Epoch  40  loss  3.6396732735251685 correct 45
Time taken for 10 epochs:  17.942656993865967
Epoch  50  loss  2.429762827508566 correct 49
Time taken for 10 epochs:  14.361004114151001

GPU, stopped after correct >= npts -1, time in seconds, had to stop early for gpu limit
Epoch  0  loss  9.20761171001891 correct 24
Time taken for 10 epochs:  1.1156349182128906
Epoch  10  loss  5.937320658188506 correct 30
Time taken for 10 epochs:  1.1232678890228271
Epoch  20  loss  4.11666184015867 correct 31
Time taken for 10 epochs:  1.1404433250427246
Epoch  30  loss  3.9619041152323184 correct 47
Time taken for 10 epochs:  1.130012035369873
Epoch  40  loss  3.6561611415717583 correct 46
Time taken for 10 epochs:  1.117152214050293
Epoch  50  loss  2.870298191527188 correct 49

Dataset: xor, hidden 100, lr: 0.05
CPU, stopped after correct >= npts -1, time in seconds
Epoch  0  loss  8.050857813331666 correct 28
Time taken for 10 epochs:  1.1397578716278076
Epoch  10  loss  4.16068033710834 correct 40
Time taken for 10 epochs:  1.1225204467773438
Epoch  20  loss  4.706509380930873 correct 42
Time taken for 10 epochs:  1.6341824531555176
Epoch  30  loss  4.53840319937567 correct 41
Time taken for 10 epochs:  1.7433452606201172
Epoch  40  loss  4.691104052306848 correct 46
Time taken for 10 epochs:  1.114884853363037
Epoch  50  loss  2.4473043456183907 correct 46
Time taken for 10 epochs:  1.1255204677581787
Epoch  60  loss  3.3411526792946633 correct 46
Time taken for 10 epochs:  1.1148602962493896
Epoch  70  loss  2.9338010665774585 correct 47
Time taken for 10 epochs:  1.131582260131836
Epoch  80  loss  1.8800748788338932 correct 46
Time taken for 10 epochs:  1.1408491134643555
Epoch  90  loss  1.8556419317483055 correct 48
Time taken for 10 epochs:  1.1298322677612305
Epoch  100  loss  2.8759015518066753 correct 47
Time taken for 10 epochs:  1.139786720275879
Epoch  110  loss  1.7901981290356679 correct 45
Time taken for 10 epochs:  1.1233394145965576
Epoch  120  loss  1.68915699626208 correct 47
Time taken for 10 epochs:  1.5419068336486816
Epoch  130  loss  2.6820187714451236 correct 48
Time taken for 10 epochs:  1.8337185382843018
Epoch  140  loss  1.604400771747039 correct 48
Time taken for 10 epochs:  1.1126117706298828
Epoch  150  loss  0.6298951482267238 correct 47
Time taken for 10 epochs:  1.1127994060516357
Epoch  160  loss  1.4762706468852818 correct 48
Time taken for 10 epochs:  1.1494598388671875
Epoch  170  loss  0.77669304767523 correct 48
Time taken for 10 epochs:  1.1309027671813965
Epoch  180  loss  2.0689689050228193 correct 48
Time taken for 10 epochs:  1.1332612037658691
Epoch  190  loss  1.7354240395760274 correct 49

GPU, stopped after correct >= npts -1, time in seconds
Epoch  0  loss  7.493458559631764 correct 28
Time taken for 10 epochs:  18.69411873817444
Epoch  10  loss  4.485812530224038 correct 35
Time taken for 10 epochs:  17.035502672195435
Epoch  20  loss  3.689602493964667 correct 42
Time taken for 10 epochs:  17.150625705718994
Epoch  30  loss  3.5928731569590875 correct 42
Time taken for 10 epochs:  17.785924196243286
Epoch  40  loss  3.222000247514872 correct 39
Time taken for 10 epochs:  16.98338770866394
Epoch  50  loss  2.980923097752376 correct 42
Time taken for 10 epochs:  17.8644118309021
Epoch  60  loss  2.147236793970005 correct 43
Time taken for 10 epochs:  16.92519736289978
Epoch  70  loss  2.3104796515441155 correct 43
Time taken for 10 epochs:  17.092494010925293
Epoch  80  loss  4.165683474937243 correct 48
Time taken for 10 epochs:  17.677455186843872
Epoch  90  loss  1.2612448439818544 correct 46
Time taken for 10 epochs:  17.82852005958557
Epoch  100  loss  2.208616294848576 correct 46
Time taken for 10 epochs:  17.899542808532715
Epoch  110  loss  1.0694638146899058 correct 48
Time taken for 10 epochs:  16.964478015899658
Epoch  120  loss  3.4258269205082033 correct 47
Time taken for 10 epochs:  17.505073308944702
Epoch  130  loss  2.020547491787324 correct 48
Time taken for 10 epochs:  17.037348747253418
Epoch  140  loss  1.7103794504279972 correct 48
Time taken for 10 epochs:  16.886613845825195
Epoch  150  loss  1.4559540870457455 correct 49
Time taken for 10 epochs:  14.228630781173706

Numba parallel check

MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/schade/workspace/mod3-peterfschade/minitorch/fast_ops.py (164)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/schade/workspace/mod3-peterfschade/minitorch/fast_ops.py (164)
--------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                         |
        out: Storage,                                                                 |
        out_shape: Shape,                                                             |
        out_strides: Strides,                                                         |
        in_storage: Storage,                                                          |
        in_shape: Shape,                                                              |
        in_strides: Strides,                                                          |
    ) -> None:                                                                        |
        # TODO: Implement for Task 3.1.                                               |
                                                                                      |
        stride_aligned = 0                                                            |
        if len(out_shape) == len(in_shape):                                           |
            stride_aligned = 1                                                        |
            for i in range(len(out_shape)):                                           |
                if out_shape[i] != in_shape[i] or out_strides[i] != in_strides[i]:    |
                    stride_aligned = 0                                                |
                    break                                                             |
                                                                                      |
        for i in prange(len(out)):----------------------------------------------------| #0
            if stride_aligned > 0:                                                    |
                out[i] = fn(in_storage[i])                                            |
            else:                                                                     |
                out_idx: Index = np.empty(MAX_DIMS, np.int32)                         |
                in_idx: Index = np.empty(MAX_DIMS, np.int32)                          |
                to_index(i, out_shape, out_idx)                                       |
                broadcast_index(out_idx, out_shape, in_shape, in_idx)                 |
                o = index_to_position(out_idx, out_strides)                           |
                j = index_to_position(in_idx, in_strides)                             |
                out[o] = fn(in_storage[j])                                            |
        # raise NotImplementedError("Need to implement for Task 3.1")                 |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #0).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/schade/workspace/mod3-peterfschade/minitorch/fast_ops.py (186) is hoisted
 out of the parallel loop labelled #0 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_idx: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/schade/workspace/mod3-peterfschade/minitorch/fast_ops.py (187) is hoisted
 out of the parallel loop labelled #0 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: in_idx: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/schade/workspace/mod3-peterfschade/minitorch/fast_ops.py (221)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/schade/workspace/mod3-peterfschade/minitorch/fast_ops.py (221)
--------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                         |
        out: Storage,                                                                 |
        out_shape: Shape,                                                             |
        out_strides: Strides,                                                         |
        a_storage: Storage,                                                           |
        a_shape: Shape,                                                               |
        a_strides: Strides,                                                           |
        b_storage: Storage,                                                           |
        b_shape: Shape,                                                               |
        b_strides: Strides,                                                           |
    ) -> None:                                                                        |
        # TODO: Implement for Task 3.1.                                               |
                                                                                      |
        stride_aligned = 0                                                            |
        if len(out_shape) == len(a_shape) & len(a_shape) == len(b_shape):             |
            stride_aligned = 1                                                        |
                                                                                      |
            for i in range(len(out_shape)):                                           |
                if out_shape[i] != a_shape[i] or out_strides[i] != a_strides[i]:      |
                    stride_aligned = 0                                                |
                    break                                                             |
                elif out_shape[i] != b_shape[i] or out_strides[i] != b_strides[i]:    |
                    stride_aligned = 0                                                |
                    break                                                             |
                                                                                      |
        for i in prange(len(out)):  # main loop---------------------------------------| #1
            if stride_aligned > 0:                                                    |
                out[i] = fn(a_storage[i], b_storage[i])                               |
            else:                                                                     |
                out_idx: Index = np.empty(MAX_DIMS, np.int32)                         |
                a_idx: Index = np.empty(MAX_DIMS, np.int32)                           |
                b_idx: Index = np.empty(MAX_DIMS, np.int32)                           |
                to_index(i, out_shape, out_idx)                                       |
                o = index_to_position(out_idx, out_strides)                           |
                broadcast_index(out_idx, out_shape, a_shape, a_idx)                   |
                j = index_to_position(a_idx, a_strides)                               |
                broadcast_index(out_idx, out_shape, b_shape, b_idx)                   |
                k = index_to_position(b_idx, b_strides)                               |
                out[o] = fn(a_storage[j], b_storage[k])                               |
        # raise NotImplementedError("Need to implement for Task 3.1")                 |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/schade/workspace/mod3-peterfschade/minitorch/fast_ops.py (250) is hoisted
 out of the parallel loop labelled #1 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_idx: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/schade/workspace/mod3-peterfschade/minitorch/fast_ops.py (251) is hoisted
 out of the parallel loop labelled #1 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: a_idx: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/schade/workspace/mod3-peterfschade/minitorch/fast_ops.py (252) is hoisted
 out of the parallel loop labelled #1 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: b_idx: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/schade/workspace/mod3-peterfschade/minitorch/fast_ops.py (286)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/schade/workspace/mod3-peterfschade/minitorch/fast_ops.py (286)
-------------------------------------------------------------------------|loop #ID
    def _reduce(                                                         |
        out: Storage,                                                    |
        out_shape: Shape,                                                |
        out_strides: Strides,                                            |
        a_storage: Storage,                                              |
        a_shape: Shape,                                                  |
        a_strides: Strides,                                              |
        reduce_dim: int,                                                 |
    ) -> None:                                                           |
        # TODO: Implement for Task 3.1.                                  |
                                                                         |
        reduce_size = a_shape[reduce_dim]                                |
                                                                         |
        for i in prange(len(out)):---------------------------------------| #2
            out_idx: Index = np.empty(MAX_DIMS, np.int32)                |
            to_index(i, out_shape, out_idx)                              |
            o_pos = index_to_position(out_idx, out_strides)              |
            k0 = index_to_position(out_idx, a_strides)                   |
            t = a_storage[k0]                                            |
                                                                         |
            for j in range(1, reduce_size):                              |
                k = k0 + a_strides[reduce_dim] * j                       |
                t = fn(t, a_storage[int(k)])                             |
            out[o_pos] = t                                               |
        # raise NotImplementedError("Need to implement for Task 3.1")    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #2).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/schade/workspace/mod3-peterfschade/minitorch/fast_ops.py (300) is hoisted
 out of the parallel loop labelled #2 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_idx: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/schade/workspace/mod3-peterfschade/minitorch/fast_ops.py (315)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/schade/workspace/mod3-peterfschade/minitorch/fast_ops.py (315)
------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                              |
    out: Storage,                                                                         |
    out_shape: Shape,                                                                     |
    out_strides: Strides,                                                                 |
    a_storage: Storage,                                                                   |
    a_shape: Shape,                                                                       |
    a_strides: Strides,                                                                   |
    b_storage: Storage,                                                                   |
    b_shape: Shape,                                                                       |
    b_strides: Strides,                                                                   |
) -> None:                                                                                |
    """NUMBA tensor matrix multiply function.                                             |
                                                                                          |
    Should work for any tensor shapes that broadcast as long as                           |
                                                                                          |
    ```                                                                                   |
    assert a_shape[-1] == b_shape[-2]                                                     |
    ```                                                                                   |
                                                                                          |
    Optimizations:                                                                        |
                                                                                          |
    * Outer loop in parallel                                                              |
    * No index buffers or function calls                                                  |
    * Inner loop should have no global writes, 1 multiply.                                |
                                                                                          |
                                                                                          |
    Args:                                                                                 |
    ----                                                                                  |
        out (Storage): storage for `out` tensor                                           |
        out_shape (Shape): shape for `out` tensor                                         |
        out_strides (Strides): strides for `out` tensor                                   |
        a_storage (Storage): storage for `a` tensor                                       |
        a_shape (Shape): shape for `a` tensor                                             |
        a_strides (Strides): strides for `a` tensor                                       |
        b_storage (Storage): storage for `b` tensor                                       |
        b_shape (Shape): shape for `b` tensor                                             |
        b_strides (Strides): strides for `b` tensor                                       |
                                                                                          |
    Returns:                                                                              |
    -------                                                                               |
        None : Fills in `out`                                                             |
                                                                                          |
    """                                                                                   |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                |
                                                                                          |
    assert a_shape[-1] == b_shape[-2], "Cannot matmul these tensors, dims don't match"    |
                                                                                          |
    # for outer_index in out.indices():                                                   |
    #    for inner_val in range(J):                                                       |
    #        out[outer_index] += A[outer_index[0], inner_val] * \                         |
    #                            B[inner_val, outer_index[1]]                             |
                                                                                          |
    for i in prange(len(out)):------------------------------------------------------------| #3
        out_0 = i // (out_shape[-1] * out_shape[-2])                                      |
        out_1 = (i % (out_shape[-1] * out_shape[-2])) // out_shape[-1]                    |
        out_2 = i % out_shape[-1]                                                         |
                                                                                          |
        a_ = out_0 * a_batch_stride + out_1 * a_strides[-2]                               |
        b_ = out_0 * b_batch_stride + out_2 * b_strides[-1]                               |
                                                                                          |
        t = 0                                                                             |
        for p in range(a_shape[-1]):                                                      |
            t += a_storage[a_ + p * a_strides[-1]] * b_storage[b_ + p * b_strides[-2]]    |
        out[i] = t                                                                        |
                                                                                          |
    # TODO: Implement for Task 3.2.                                                       |
    # raise NotImplementedError("Need to implement for Task 3.2")                         |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

