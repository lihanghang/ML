��
�0�/
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

ControlTrigger
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
:
Less
x"T
y"T
z
"
Ttype:
2	
$

LogicalAnd
x

y

z
�
!
LoopCond	
input


output

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
;
Minimum
x"T
y"T
z"T"
Ttype:

2	�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
1
Square
x"T
y"T"
Ttype:

2	
A

StackPopV2

handle
elem"	elem_type"
	elem_typetype�
X
StackPushV2

handle	
elem"T
output"T"	
Ttype"
swap_memorybool( �
S
StackV2
max_size

handle"
	elem_typetype"

stack_namestring �
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:�
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestring�
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype�
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype�
9
TensorArraySizeV3

handle
flow_in
size�
�
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring �
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype�
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �"mytag*1.10.02
b'unknown'ʁ
q
inputsPlaceholder* 
shape:���������*
dtype0*+
_output_shapes
:���������
r
outputsPlaceholder* 
shape:���������*
dtype0*+
_output_shapes
:���������
r
!layer/weights/random_normal/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
e
 layer/weights/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
g
"layer/weights/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
0layer/weights/random_normal/RandomStandardNormalRandomStandardNormal!layer/weights/random_normal/shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
layer/weights/random_normal/mulMul0layer/weights/random_normal/RandomStandardNormal"layer/weights/random_normal/stddev*
T0*
_output_shapes

:

�
layer/weights/random_normalAddlayer/weights/random_normal/mul layer/weights/random_normal/mean*
T0*
_output_shapes

:

�
layer/weights/Variable
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
layer/weights/Variable/AssignAssignlayer/weights/Variablelayer/weights/random_normal*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
layer/weights/Variable/readIdentitylayer/weights/Variable*
T0*)
_class
loc:@layer/weights/Variable*
_output_shapes

:

t
#layer/weights/random_normal_1/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
g
"layer/weights/random_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
$layer/weights/random_normal_1/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
2layer/weights/random_normal_1/RandomStandardNormalRandomStandardNormal#layer/weights/random_normal_1/shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
!layer/weights/random_normal_1/mulMul2layer/weights/random_normal_1/RandomStandardNormal$layer/weights/random_normal_1/stddev*
T0*
_output_shapes

:

�
layer/weights/random_normal_1Add!layer/weights/random_normal_1/mul"layer/weights/random_normal_1/mean*
T0*
_output_shapes

:

�
layer/weights/Variable_1
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
layer/weights/Variable_1/AssignAssignlayer/weights/Variable_1layer/weights/random_normal_1*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
layer/weights/Variable_1/readIdentitylayer/weights/Variable_1*
T0*+
_class!
loc:@layer/weights/Variable_1*
_output_shapes

:

_
layer/biases/ConstConst*
valueB
*���=*
dtype0*
_output_shapes
:

�
layer/biases/Variable
VariableV2*
dtype0*
_output_shapes
:
*
	container *
shape:
*
shared_name 
�
layer/biases/Variable/AssignAssignlayer/biases/Variablelayer/biases/Const*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

�
layer/biases/Variable/readIdentitylayer/biases/Variable*
T0*(
_class
loc:@layer/biases/Variable*
_output_shapes
:

a
layer/biases/Const_1Const*
valueB*���=*
dtype0*
_output_shapes
:
�
layer/biases/Variable_1
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
layer/biases/Variable_1/AssignAssignlayer/biases/Variable_1layer/biases/Const_1*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
layer/biases/Variable_1/readIdentitylayer/biases/Variable_1*
T0**
_class 
loc:@layer/biases/Variable_1*
_output_shapes
:
b
rnn/Reshape/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
q
rnn/ReshapeReshapeinputsrnn/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:���������
�

rnn/MatMulMatMulrnn/Reshapelayer/weights/Variable/read*
transpose_b( *
T0*'
_output_shapes
:���������
*
transpose_a( 
h
rnn/addAdd
rnn/MatMullayer/biases/Variable/read*
T0*'
_output_shapes
:���������

h
rnn/Reshape_1/shapeConst*!
valueB"����   
   *
dtype0*
_output_shapes
:
z
rnn/Reshape_1Reshapernn/addrnn/Reshape_1/shape*
T0*
Tshape0*+
_output_shapes
:���������

j
 rnn/BasicLSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
l
"rnn/BasicLSTMCellZeroState/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
h
&rnn/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
!rnn/BasicLSTMCellZeroState/concatConcatV2 rnn/BasicLSTMCellZeroState/Const"rnn/BasicLSTMCellZeroState/Const_1&rnn/BasicLSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
k
&rnn/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 rnn/BasicLSTMCellZeroState/zerosFill!rnn/BasicLSTMCellZeroState/concat&rnn/BasicLSTMCellZeroState/zeros/Const*
T0*

index_type0*
_output_shapes

:

l
"rnn/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
l
"rnn/BasicLSTMCellZeroState/Const_3Const*
valueB:
*
dtype0*
_output_shapes
:
l
"rnn/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
l
"rnn/BasicLSTMCellZeroState/Const_5Const*
valueB:
*
dtype0*
_output_shapes
:
j
(rnn/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
#rnn/BasicLSTMCellZeroState/concat_1ConcatV2"rnn/BasicLSTMCellZeroState/Const_4"rnn/BasicLSTMCellZeroState/Const_5(rnn/BasicLSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
m
(rnn/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"rnn/BasicLSTMCellZeroState/zeros_1Fill#rnn/BasicLSTMCellZeroState/concat_1(rnn/BasicLSTMCellZeroState/zeros_1/Const*
T0*

index_type0*
_output_shapes

:

l
"rnn/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
l
"rnn/BasicLSTMCellZeroState/Const_7Const*
valueB:
*
dtype0*
_output_shapes
:
N
rnn/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
U
rnn/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
U
rnn/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
v
rnn/rnn/rangeRangernn/rnn/range/startrnn/rnn/Rankrnn/rnn/range/delta*

Tidx0*
_output_shapes
:
h
rnn/rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
U
rnn/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn/rnn/concatConcatV2rnn/rnn/concat/values_0rnn/rnn/rangernn/rnn/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
rnn/rnn/transpose	Transposernn/Reshape_1rnn/rnn/concat*
T0*+
_output_shapes
:���������
*
Tperm0
^
rnn/rnn/ShapeShapernn/rnn/transpose*
T0*
out_type0*
_output_shapes
:
e
rnn/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
g
rnn/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
g
rnn/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
rnn/rnn/strided_sliceStridedSlicernn/rnn/Shapernn/rnn/strided_slice/stackrnn/rnn/strided_slice/stack_1rnn/rnn/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
`
rnn/rnn/Shape_1Shapernn/rnn/transpose*
T0*
out_type0*
_output_shapes
:
g
rnn/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
i
rnn/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
i
rnn/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
rnn/rnn/strided_slice_1StridedSlicernn/rnn/Shape_1rnn/rnn/strided_slice_1/stackrnn/rnn/strided_slice_1/stack_1rnn/rnn/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
`
rnn/rnn/Shape_2Shapernn/rnn/transpose*
T0*
out_type0*
_output_shapes
:
g
rnn/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
rnn/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
i
rnn/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
rnn/rnn/strided_slice_2StridedSlicernn/rnn/Shape_2rnn/rnn/strided_slice_2/stackrnn/rnn/strided_slice_2/stack_1rnn/rnn/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
X
rnn/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn/rnn/ExpandDims
ExpandDimsrnn/rnn/strided_slice_2rnn/rnn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
W
rnn/rnn/ConstConst*
valueB:
*
dtype0*
_output_shapes
:
W
rnn/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn/rnn/concat_1ConcatV2rnn/rnn/ExpandDimsrnn/rnn/Constrnn/rnn/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
X
rnn/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
rnn/rnn/zerosFillrnn/rnn/concat_1rnn/rnn/zeros/Const*
T0*

index_type0*'
_output_shapes
:���������

N
rnn/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn/rnn/TensorArrayTensorArrayV3rnn/rnn/strided_slice_1*3
tensor_array_namernn/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *$
element_shape:���������
*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
�
rnn/rnn/TensorArray_1TensorArrayV3rnn/rnn/strided_slice_1*$
element_shape:���������
*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*2
tensor_array_namernn/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: 
q
 rnn/rnn/TensorArrayUnstack/ShapeShapernn/rnn/transpose*
T0*
out_type0*
_output_shapes
:
x
.rnn/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0rnn/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0rnn/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
(rnn/rnn/TensorArrayUnstack/strided_sliceStridedSlice rnn/rnn/TensorArrayUnstack/Shape.rnn/rnn/TensorArrayUnstack/strided_slice/stack0rnn/rnn/TensorArrayUnstack/strided_slice/stack_10rnn/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
h
&rnn/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
h
&rnn/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
 rnn/rnn/TensorArrayUnstack/rangeRange&rnn/rnn/TensorArrayUnstack/range/start(rnn/rnn/TensorArrayUnstack/strided_slice&rnn/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:���������*

Tidx0
�
Brnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn/rnn/TensorArray_1 rnn/rnn/TensorArrayUnstack/rangernn/rnn/transposernn/rnn/TensorArray_1:1*
T0*$
_class
loc:@rnn/rnn/transpose*
_output_shapes
: 
S
rnn/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
g
rnn/rnn/MaximumMaximumrnn/rnn/Maximum/xrnn/rnn/strided_slice_1*
T0*
_output_shapes
: 
e
rnn/rnn/MinimumMinimumrnn/rnn/strided_slice_1rnn/rnn/Maximum*
T0*
_output_shapes
: 
a
rnn/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn/rnn/while/EnterEnterrnn/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *+

frame_namernn/rnn/while/while_context
�
rnn/rnn/while/Enter_1Enterrnn/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *+

frame_namernn/rnn/while/while_context
�
rnn/rnn/while/Enter_2Enterrnn/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *+

frame_namernn/rnn/while/while_context
�
rnn/rnn/while/Enter_3Enter rnn/BasicLSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*+

frame_namernn/rnn/while/while_context
�
rnn/rnn/while/Enter_4Enter"rnn/BasicLSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*+

frame_namernn/rnn/while/while_context
z
rnn/rnn/while/MergeMergernn/rnn/while/Enterrnn/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 
�
rnn/rnn/while/Merge_1Mergernn/rnn/while/Enter_1rnn/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
rnn/rnn/while/Merge_2Mergernn/rnn/while/Enter_2rnn/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
�
rnn/rnn/while/Merge_3Mergernn/rnn/while/Enter_3rnn/rnn/while/NextIteration_3*
T0*
N* 
_output_shapes
:
: 
�
rnn/rnn/while/Merge_4Mergernn/rnn/while/Enter_4rnn/rnn/while/NextIteration_4*
T0*
N* 
_output_shapes
:
: 
j
rnn/rnn/while/LessLessrnn/rnn/while/Mergernn/rnn/while/Less/Enter*
T0*
_output_shapes
: 
�
rnn/rnn/while/Less/EnterEnterrnn/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *+

frame_namernn/rnn/while/while_context
p
rnn/rnn/while/Less_1Lessrnn/rnn/while/Merge_1rnn/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
�
rnn/rnn/while/Less_1/EnterEnterrnn/rnn/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *+

frame_namernn/rnn/while/while_context
h
rnn/rnn/while/LogicalAnd
LogicalAndrnn/rnn/while/Lessrnn/rnn/while/Less_1*
_output_shapes
: 
T
rnn/rnn/while/LoopCondLoopCondrnn/rnn/while/LogicalAnd*
_output_shapes
: 
�
rnn/rnn/while/SwitchSwitchrnn/rnn/while/Mergernn/rnn/while/LoopCond*
T0*&
_class
loc:@rnn/rnn/while/Merge*
_output_shapes
: : 
�
rnn/rnn/while/Switch_1Switchrnn/rnn/while/Merge_1rnn/rnn/while/LoopCond*
T0*(
_class
loc:@rnn/rnn/while/Merge_1*
_output_shapes
: : 
�
rnn/rnn/while/Switch_2Switchrnn/rnn/while/Merge_2rnn/rnn/while/LoopCond*
T0*(
_class
loc:@rnn/rnn/while/Merge_2*
_output_shapes
: : 
�
rnn/rnn/while/Switch_3Switchrnn/rnn/while/Merge_3rnn/rnn/while/LoopCond*
T0*(
_class
loc:@rnn/rnn/while/Merge_3*(
_output_shapes
:
:

�
rnn/rnn/while/Switch_4Switchrnn/rnn/while/Merge_4rnn/rnn/while/LoopCond*
T0*(
_class
loc:@rnn/rnn/while/Merge_4*(
_output_shapes
:
:

[
rnn/rnn/while/IdentityIdentityrnn/rnn/while/Switch:1*
T0*
_output_shapes
: 
_
rnn/rnn/while/Identity_1Identityrnn/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
_
rnn/rnn/while/Identity_2Identityrnn/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
g
rnn/rnn/while/Identity_3Identityrnn/rnn/while/Switch_3:1*
T0*
_output_shapes

:

g
rnn/rnn/while/Identity_4Identityrnn/rnn/while/Switch_4:1*
T0*
_output_shapes

:

n
rnn/rnn/while/add/yConst^rnn/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
f
rnn/rnn/while/addAddrnn/rnn/while/Identityrnn/rnn/while/add/y*
T0*
_output_shapes
: 
�
rnn/rnn/while/TensorArrayReadV3TensorArrayReadV3%rnn/rnn/while/TensorArrayReadV3/Enterrnn/rnn/while/Identity_1'rnn/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:���������

�
%rnn/rnn/while/TensorArrayReadV3/EnterEnterrnn/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*+

frame_namernn/rnn/while/while_context
�
'rnn/rnn/while/TensorArrayReadV3/Enter_1EnterBrnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *+

frame_namernn/rnn/while/while_context
�
?rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
valueB"   (   *
dtype0*
_output_shapes
:
�
=rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
valueB
 *�衾*
dtype0*
_output_shapes
: 
�
=rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
valueB
 *��>*
dtype0*
_output_shapes
: 
�
Grnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform?rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
seed2 *
dtype0*
_output_shapes

:(*

seed 
�
=rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/subSub=rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/max=rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
_output_shapes
: 
�
=rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulGrnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniform=rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
9rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniformAdd=rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mul=rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
rnn/rnn/basic_lstm_cell/kernel
VariableV2*
shape
:(*
dtype0*
_output_shapes

:(*
shared_name *1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
	container 
�
%rnn/rnn/basic_lstm_cell/kernel/AssignAssignrnn/rnn/basic_lstm_cell/kernel9rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
x
#rnn/rnn/basic_lstm_cell/kernel/readIdentityrnn/rnn/basic_lstm_cell/kernel*
T0*
_output_shapes

:(
�
.rnn/rnn/basic_lstm_cell/bias/Initializer/zerosConst*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
valueB(*    *
dtype0*
_output_shapes
:(
�
rnn/rnn/basic_lstm_cell/bias
VariableV2*
dtype0*
_output_shapes
:(*
shared_name */
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
	container *
shape:(
�
#rnn/rnn/basic_lstm_cell/bias/AssignAssignrnn/rnn/basic_lstm_cell/bias.rnn/rnn/basic_lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
p
!rnn/rnn/basic_lstm_cell/bias/readIdentityrnn/rnn/basic_lstm_cell/bias*
T0*
_output_shapes
:(
~
#rnn/rnn/while/basic_lstm_cell/ConstConst^rnn/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
)rnn/rnn/while/basic_lstm_cell/concat/axisConst^rnn/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
$rnn/rnn/while/basic_lstm_cell/concatConcatV2rnn/rnn/while/TensorArrayReadV3rnn/rnn/while/Identity_4)rnn/rnn/while/basic_lstm_cell/concat/axis*
T0*
N*
_output_shapes

:*

Tidx0
�
$rnn/rnn/while/basic_lstm_cell/MatMulMatMul$rnn/rnn/while/basic_lstm_cell/concat*rnn/rnn/while/basic_lstm_cell/MatMul/Enter*
T0*
_output_shapes

:(*
transpose_a( *
transpose_b( 
�
*rnn/rnn/while/basic_lstm_cell/MatMul/EnterEnter#rnn/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*+

frame_namernn/rnn/while/while_context
�
%rnn/rnn/while/basic_lstm_cell/BiasAddBiasAdd$rnn/rnn/while/basic_lstm_cell/MatMul+rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes

:(
�
+rnn/rnn/while/basic_lstm_cell/BiasAdd/EnterEnter!rnn/rnn/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:(*+

frame_namernn/rnn/while/while_context
�
%rnn/rnn/while/basic_lstm_cell/Const_1Const^rnn/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
#rnn/rnn/while/basic_lstm_cell/splitSplit#rnn/rnn/while/basic_lstm_cell/Const%rnn/rnn/while/basic_lstm_cell/BiasAdd*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split
�
%rnn/rnn/while/basic_lstm_cell/Const_2Const^rnn/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
!rnn/rnn/while/basic_lstm_cell/AddAdd%rnn/rnn/while/basic_lstm_cell/split:2%rnn/rnn/while/basic_lstm_cell/Const_2*
T0*
_output_shapes

:

|
%rnn/rnn/while/basic_lstm_cell/SigmoidSigmoid!rnn/rnn/while/basic_lstm_cell/Add*
T0*
_output_shapes

:

�
!rnn/rnn/while/basic_lstm_cell/MulMulrnn/rnn/while/Identity_3%rnn/rnn/while/basic_lstm_cell/Sigmoid*
T0*
_output_shapes

:

�
'rnn/rnn/while/basic_lstm_cell/Sigmoid_1Sigmoid#rnn/rnn/while/basic_lstm_cell/split*
T0*
_output_shapes

:

z
"rnn/rnn/while/basic_lstm_cell/TanhTanh%rnn/rnn/while/basic_lstm_cell/split:1*
T0*
_output_shapes

:

�
#rnn/rnn/while/basic_lstm_cell/Mul_1Mul'rnn/rnn/while/basic_lstm_cell/Sigmoid_1"rnn/rnn/while/basic_lstm_cell/Tanh*
T0*
_output_shapes

:

�
#rnn/rnn/while/basic_lstm_cell/Add_1Add!rnn/rnn/while/basic_lstm_cell/Mul#rnn/rnn/while/basic_lstm_cell/Mul_1*
T0*
_output_shapes

:

z
$rnn/rnn/while/basic_lstm_cell/Tanh_1Tanh#rnn/rnn/while/basic_lstm_cell/Add_1*
T0*
_output_shapes

:

�
'rnn/rnn/while/basic_lstm_cell/Sigmoid_2Sigmoid%rnn/rnn/while/basic_lstm_cell/split:3*
T0*
_output_shapes

:

�
#rnn/rnn/while/basic_lstm_cell/Mul_2Mul$rnn/rnn/while/basic_lstm_cell/Tanh_1'rnn/rnn/while/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes

:

�
1rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV37rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn/rnn/while/Identity_1#rnn/rnn/while/basic_lstm_cell/Mul_2rnn/rnn/while/Identity_2*
T0*6
_class,
*(loc:@rnn/rnn/while/basic_lstm_cell/Mul_2*
_output_shapes
: 
�
7rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn/rnn/TensorArray*
T0*6
_class,
*(loc:@rnn/rnn/while/basic_lstm_cell/Mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*+

frame_namernn/rnn/while/while_context
p
rnn/rnn/while/add_1/yConst^rnn/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
l
rnn/rnn/while/add_1Addrnn/rnn/while/Identity_1rnn/rnn/while/add_1/y*
T0*
_output_shapes
: 
`
rnn/rnn/while/NextIterationNextIterationrnn/rnn/while/add*
T0*
_output_shapes
: 
d
rnn/rnn/while/NextIteration_1NextIterationrnn/rnn/while/add_1*
T0*
_output_shapes
: 
�
rnn/rnn/while/NextIteration_2NextIteration1rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
|
rnn/rnn/while/NextIteration_3NextIteration#rnn/rnn/while/basic_lstm_cell/Add_1*
T0*
_output_shapes

:

|
rnn/rnn/while/NextIteration_4NextIteration#rnn/rnn/while/basic_lstm_cell/Mul_2*
T0*
_output_shapes

:

Q
rnn/rnn/while/ExitExitrnn/rnn/while/Switch*
T0*
_output_shapes
: 
U
rnn/rnn/while/Exit_1Exitrnn/rnn/while/Switch_1*
T0*
_output_shapes
: 
U
rnn/rnn/while/Exit_2Exitrnn/rnn/while/Switch_2*
T0*
_output_shapes
: 
]
rnn/rnn/while/Exit_3Exitrnn/rnn/while/Switch_3*
T0*
_output_shapes

:

]
rnn/rnn/while/Exit_4Exitrnn/rnn/while/Switch_4*
T0*
_output_shapes

:

�
*rnn/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn/rnn/TensorArrayrnn/rnn/while/Exit_2*&
_class
loc:@rnn/rnn/TensorArray*
_output_shapes
: 
�
$rnn/rnn/TensorArrayStack/range/startConst*&
_class
loc:@rnn/rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 
�
$rnn/rnn/TensorArrayStack/range/deltaConst*&
_class
loc:@rnn/rnn/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
�
rnn/rnn/TensorArrayStack/rangeRange$rnn/rnn/TensorArrayStack/range/start*rnn/rnn/TensorArrayStack/TensorArraySizeV3$rnn/rnn/TensorArrayStack/range/delta*&
_class
loc:@rnn/rnn/TensorArray*#
_output_shapes
:���������*

Tidx0
�
,rnn/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn/rnn/TensorArrayrnn/rnn/TensorArrayStack/rangernn/rnn/while/Exit_2*
element_shape
:
*&
_class
loc:@rnn/rnn/TensorArray*
dtype0*"
_output_shapes
:

Y
rnn/rnn/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
P
rnn/rnn/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
W
rnn/rnn/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
W
rnn/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
~
rnn/rnn/range_1Rangernn/rnn/range_1/startrnn/rnn/Rank_1rnn/rnn/range_1/delta*

Tidx0*
_output_shapes
:
j
rnn/rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
W
rnn/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn/rnn/concat_2ConcatV2rnn/rnn/concat_2/values_0rnn/rnn/range_1rnn/rnn/concat_2/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
rnn/rnn/transpose_1	Transpose,rnn/rnn/TensorArrayStack/TensorArrayGatherV3rnn/rnn/concat_2*
T0*"
_output_shapes
:
*
Tperm0
d
rnn/Reshape_2/shapeConst*
valueB"����
   *
dtype0*
_output_shapes
:
y
rnn/Reshape_2Reshapernn/rnn/transpose_1rnn/Reshape_2/shape*
T0*
Tshape0*
_output_shapes

:


�
rnn/MatMul_1MatMulrnn/Reshape_2layer/weights/Variable_1/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b( 
e
	rnn/predsAddrnn/MatMul_1layer/biases/Variable_1/read*
T0*
_output_shapes

:

f
rnn/Reshape_3/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
k
rnn/Reshape_3Reshape	rnn/predsrnn/Reshape_3/shape*
T0*
Tshape0*
_output_shapes
:

f
rnn/Reshape_4/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
r
rnn/Reshape_4Reshapeoutputsrnn/Reshape_4/shape*
T0*
Tshape0*#
_output_shapes
:���������
Q
rnn/subSubrnn/Reshape_3rnn/Reshape_4*
T0*
_output_shapes
:

B

rnn/SquareSquarernn/sub*
T0*
_output_shapes
:

S
	rnn/ConstConst*
valueB: *
dtype0*
_output_shapes
:
e
rnn/MeanMean
rnn/Square	rnn/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
V
rnn/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
\
rnn/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
{
rnn/gradients/FillFillrnn/gradients/Shapernn/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
W
rnn/gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn/gradients/f_count_1Enterrnn/gradients/f_count*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *+

frame_namernn/rnn/while/while_context
~
rnn/gradients/MergeMergernn/gradients/f_count_1rnn/gradients/NextIteration*
T0*
N*
_output_shapes
: : 
n
rnn/gradients/SwitchSwitchrnn/gradients/Mergernn/rnn/while/LoopCond*
T0*
_output_shapes
: : 
n
rnn/gradients/Add/yConst^rnn/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
f
rnn/gradients/AddAddrnn/gradients/Switch:1rnn/gradients/Add/y*
T0*
_output_shapes
: 
�
rnn/gradients/NextIterationNextIterationrnn/gradients/Addc^rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2M^rnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2G^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2I^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2G^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2I^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2E^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2G^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2K^rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2M^rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1*
T0*
_output_shapes
: 
V
rnn/gradients/f_count_2Exitrnn/gradients/Switch*
T0*
_output_shapes
: 
W
rnn/gradients/b_countConst*
value	B :*
dtype0*
_output_shapes
: 
�
rnn/gradients/b_count_1Enterrnn/gradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
rnn/gradients/Merge_1Mergernn/gradients/b_count_1rnn/gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
rnn/gradients/GreaterEqualGreaterEqualrnn/gradients/Merge_1 rnn/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
�
 rnn/gradients/GreaterEqual/EnterEnterrnn/gradients/b_count*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
W
rnn/gradients/b_count_2LoopCondrnn/gradients/GreaterEqual*
_output_shapes
: 
s
rnn/gradients/Switch_1Switchrnn/gradients/Merge_1rnn/gradients/b_count_2*
T0*
_output_shapes
: : 
u
rnn/gradients/SubSubrnn/gradients/Switch_1:1 rnn/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
�
rnn/gradients/NextIteration_1NextIterationrnn/gradients/Sub^^rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
X
rnn/gradients/b_count_3Exitrnn/gradients/Switch_1*
T0*
_output_shapes
: 
s
)rnn/gradients/rnn/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
#rnn/gradients/rnn/Mean_grad/ReshapeReshapernn/gradients/Fill)rnn/gradients/rnn/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
k
!rnn/gradients/rnn/Mean_grad/ConstConst*
valueB:
*
dtype0*
_output_shapes
:
�
 rnn/gradients/rnn/Mean_grad/TileTile#rnn/gradients/rnn/Mean_grad/Reshape!rnn/gradients/rnn/Mean_grad/Const*

Tmultiples0*
T0*
_output_shapes
:

h
#rnn/gradients/rnn/Mean_grad/Const_1Const*
valueB
 *   A*
dtype0*
_output_shapes
: 
�
#rnn/gradients/rnn/Mean_grad/truedivRealDiv rnn/gradients/rnn/Mean_grad/Tile#rnn/gradients/rnn/Mean_grad/Const_1*
T0*
_output_shapes
:

�
#rnn/gradients/rnn/Square_grad/ConstConst$^rnn/gradients/rnn/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
{
!rnn/gradients/rnn/Square_grad/MulMulrnn/sub#rnn/gradients/rnn/Square_grad/Const*
T0*
_output_shapes
:

�
#rnn/gradients/rnn/Square_grad/Mul_1Mul#rnn/gradients/rnn/Mean_grad/truediv!rnn/gradients/rnn/Square_grad/Mul*
T0*
_output_shapes
:

j
 rnn/gradients/rnn/sub_grad/ShapeConst*
valueB:
*
dtype0*
_output_shapes
:
o
"rnn/gradients/rnn/sub_grad/Shape_1Shapernn/Reshape_4*
T0*
out_type0*
_output_shapes
:
�
0rnn/gradients/rnn/sub_grad/BroadcastGradientArgsBroadcastGradientArgs rnn/gradients/rnn/sub_grad/Shape"rnn/gradients/rnn/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
rnn/gradients/rnn/sub_grad/SumSum#rnn/gradients/rnn/Square_grad/Mul_10rnn/gradients/rnn/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
"rnn/gradients/rnn/sub_grad/ReshapeReshapernn/gradients/rnn/sub_grad/Sum rnn/gradients/rnn/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:

�
 rnn/gradients/rnn/sub_grad/Sum_1Sum#rnn/gradients/rnn/Square_grad/Mul_12rnn/gradients/rnn/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
j
rnn/gradients/rnn/sub_grad/NegNeg rnn/gradients/rnn/sub_grad/Sum_1*
T0*
_output_shapes
:
�
$rnn/gradients/rnn/sub_grad/Reshape_1Reshapernn/gradients/rnn/sub_grad/Neg"rnn/gradients/rnn/sub_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������

+rnn/gradients/rnn/sub_grad/tuple/group_depsNoOp#^rnn/gradients/rnn/sub_grad/Reshape%^rnn/gradients/rnn/sub_grad/Reshape_1
�
3rnn/gradients/rnn/sub_grad/tuple/control_dependencyIdentity"rnn/gradients/rnn/sub_grad/Reshape,^rnn/gradients/rnn/sub_grad/tuple/group_deps*
T0*5
_class+
)'loc:@rnn/gradients/rnn/sub_grad/Reshape*
_output_shapes
:

�
5rnn/gradients/rnn/sub_grad/tuple/control_dependency_1Identity$rnn/gradients/rnn/sub_grad/Reshape_1,^rnn/gradients/rnn/sub_grad/tuple/group_deps*
T0*7
_class-
+)loc:@rnn/gradients/rnn/sub_grad/Reshape_1*#
_output_shapes
:���������
w
&rnn/gradients/rnn/Reshape_3_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
�
(rnn/gradients/rnn/Reshape_3_grad/ReshapeReshape3rnn/gradients/rnn/sub_grad/tuple/control_dependency&rnn/gradients/rnn/Reshape_3_grad/Shape*
T0*
Tshape0*
_output_shapes

:

s
"rnn/gradients/rnn/preds_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
n
$rnn/gradients/rnn/preds_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
2rnn/gradients/rnn/preds_grad/BroadcastGradientArgsBroadcastGradientArgs"rnn/gradients/rnn/preds_grad/Shape$rnn/gradients/rnn/preds_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
 rnn/gradients/rnn/preds_grad/SumSum(rnn/gradients/rnn/Reshape_3_grad/Reshape2rnn/gradients/rnn/preds_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

�
$rnn/gradients/rnn/preds_grad/ReshapeReshape rnn/gradients/rnn/preds_grad/Sum"rnn/gradients/rnn/preds_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
"rnn/gradients/rnn/preds_grad/Sum_1Sum(rnn/gradients/rnn/Reshape_3_grad/Reshape4rnn/gradients/rnn/preds_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
&rnn/gradients/rnn/preds_grad/Reshape_1Reshape"rnn/gradients/rnn/preds_grad/Sum_1$rnn/gradients/rnn/preds_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
-rnn/gradients/rnn/preds_grad/tuple/group_depsNoOp%^rnn/gradients/rnn/preds_grad/Reshape'^rnn/gradients/rnn/preds_grad/Reshape_1
�
5rnn/gradients/rnn/preds_grad/tuple/control_dependencyIdentity$rnn/gradients/rnn/preds_grad/Reshape.^rnn/gradients/rnn/preds_grad/tuple/group_deps*
T0*7
_class-
+)loc:@rnn/gradients/rnn/preds_grad/Reshape*
_output_shapes

:

�
7rnn/gradients/rnn/preds_grad/tuple/control_dependency_1Identity&rnn/gradients/rnn/preds_grad/Reshape_1.^rnn/gradients/rnn/preds_grad/tuple/group_deps*
T0*9
_class/
-+loc:@rnn/gradients/rnn/preds_grad/Reshape_1*
_output_shapes
:
�
&rnn/gradients/rnn/MatMul_1_grad/MatMulMatMul5rnn/gradients/rnn/preds_grad/tuple/control_dependencylayer/weights/Variable_1/read*
transpose_b(*
T0*
_output_shapes

:

*
transpose_a( 
�
(rnn/gradients/rnn/MatMul_1_grad/MatMul_1MatMulrnn/Reshape_25rnn/gradients/rnn/preds_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
�
0rnn/gradients/rnn/MatMul_1_grad/tuple/group_depsNoOp'^rnn/gradients/rnn/MatMul_1_grad/MatMul)^rnn/gradients/rnn/MatMul_1_grad/MatMul_1
�
8rnn/gradients/rnn/MatMul_1_grad/tuple/control_dependencyIdentity&rnn/gradients/rnn/MatMul_1_grad/MatMul1^rnn/gradients/rnn/MatMul_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@rnn/gradients/rnn/MatMul_1_grad/MatMul*
_output_shapes

:


�
:rnn/gradients/rnn/MatMul_1_grad/tuple/control_dependency_1Identity(rnn/gradients/rnn/MatMul_1_grad/MatMul_11^rnn/gradients/rnn/MatMul_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@rnn/gradients/rnn/MatMul_1_grad/MatMul_1*
_output_shapes

:

{
&rnn/gradients/rnn/Reshape_2_grad/ShapeConst*!
valueB"      
   *
dtype0*
_output_shapes
:
�
(rnn/gradients/rnn/Reshape_2_grad/ReshapeReshape8rnn/gradients/rnn/MatMul_1_grad/tuple/control_dependency&rnn/gradients/rnn/Reshape_2_grad/Shape*
T0*
Tshape0*"
_output_shapes
:

�
8rnn/gradients/rnn/rnn/transpose_1_grad/InvertPermutationInvertPermutationrnn/rnn/concat_2*
T0*
_output_shapes
:
�
0rnn/gradients/rnn/rnn/transpose_1_grad/transpose	Transpose(rnn/gradients/rnn/Reshape_2_grad/Reshape8rnn/gradients/rnn/rnn/transpose_1_grad/InvertPermutation*
T0*"
_output_shapes
:
*
Tperm0
�
arnn/gradients/rnn/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn/rnn/TensorArrayrnn/rnn/while/Exit_2*&
_class
loc:@rnn/rnn/TensorArray*
sourcernn/gradients*
_output_shapes

:: 
�
]rnn/gradients/rnn/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityrnn/rnn/while/Exit_2b^rnn/gradients/rnn/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*&
_class
loc:@rnn/rnn/TensorArray*
_output_shapes
: 
�
grnn/gradients/rnn/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3arnn/gradients/rnn/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3rnn/rnn/TensorArrayStack/range0rnn/gradients/rnn/rnn/transpose_1_grad/transpose]rnn/gradients/rnn/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
h
rnn/gradients/zerosConst*
valueB
*    *
dtype0*
_output_shapes

:

j
rnn/gradients/zeros_1Const*
valueB
*    *
dtype0*
_output_shapes

:

�
.rnn/gradients/rnn/rnn/while/Exit_2_grad/b_exitEntergrnn/gradients/rnn/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
.rnn/gradients/rnn/rnn/while/Exit_3_grad/b_exitEnterrnn/gradients/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
.rnn/gradients/rnn/rnn/while/Exit_4_grad/b_exitEnterrnn/gradients/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
2rnn/gradients/rnn/rnn/while/Switch_2_grad/b_switchMerge.rnn/gradients/rnn/rnn/while/Exit_2_grad/b_exit9rnn/gradients/rnn/rnn/while/Switch_2_grad_1/NextIteration*
T0*
N*
_output_shapes
: : 
�
2rnn/gradients/rnn/rnn/while/Switch_3_grad/b_switchMerge.rnn/gradients/rnn/rnn/while/Exit_3_grad/b_exit9rnn/gradients/rnn/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N* 
_output_shapes
:
: 
�
2rnn/gradients/rnn/rnn/while/Switch_4_grad/b_switchMerge.rnn/gradients/rnn/rnn/while/Exit_4_grad/b_exit9rnn/gradients/rnn/rnn/while/Switch_4_grad_1/NextIteration*
T0*
N* 
_output_shapes
:
: 
�
/rnn/gradients/rnn/rnn/while/Merge_2_grad/SwitchSwitch2rnn/gradients/rnn/rnn/while/Switch_2_grad/b_switchrnn/gradients/b_count_2*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: : 
s
9rnn/gradients/rnn/rnn/while/Merge_2_grad/tuple/group_depsNoOp0^rnn/gradients/rnn/rnn/while/Merge_2_grad/Switch
�
Arnn/gradients/rnn/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity/rnn/gradients/rnn/rnn/while/Merge_2_grad/Switch:^rnn/gradients/rnn/rnn/while/Merge_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
�
Crnn/gradients/rnn/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity1rnn/gradients/rnn/rnn/while/Merge_2_grad/Switch:1:^rnn/gradients/rnn/rnn/while/Merge_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
�
/rnn/gradients/rnn/rnn/while/Merge_3_grad/SwitchSwitch2rnn/gradients/rnn/rnn/while/Switch_3_grad/b_switchrnn/gradients/b_count_2*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_3_grad/b_switch*(
_output_shapes
:
:

s
9rnn/gradients/rnn/rnn/while/Merge_3_grad/tuple/group_depsNoOp0^rnn/gradients/rnn/rnn/while/Merge_3_grad/Switch
�
Arnn/gradients/rnn/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity/rnn/gradients/rnn/rnn/while/Merge_3_grad/Switch:^rnn/gradients/rnn/rnn/while/Merge_3_grad/tuple/group_deps*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
Crnn/gradients/rnn/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity1rnn/gradients/rnn/rnn/while/Merge_3_grad/Switch:1:^rnn/gradients/rnn/rnn/while/Merge_3_grad/tuple/group_deps*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
/rnn/gradients/rnn/rnn/while/Merge_4_grad/SwitchSwitch2rnn/gradients/rnn/rnn/while/Switch_4_grad/b_switchrnn/gradients/b_count_2*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_4_grad/b_switch*(
_output_shapes
:
:

s
9rnn/gradients/rnn/rnn/while/Merge_4_grad/tuple/group_depsNoOp0^rnn/gradients/rnn/rnn/while/Merge_4_grad/Switch
�
Arnn/gradients/rnn/rnn/while/Merge_4_grad/tuple/control_dependencyIdentity/rnn/gradients/rnn/rnn/while/Merge_4_grad/Switch:^rnn/gradients/rnn/rnn/while/Merge_4_grad/tuple/group_deps*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_4_grad/b_switch*
_output_shapes

:

�
Crnn/gradients/rnn/rnn/while/Merge_4_grad/tuple/control_dependency_1Identity1rnn/gradients/rnn/rnn/while/Merge_4_grad/Switch:1:^rnn/gradients/rnn/rnn/while/Merge_4_grad/tuple/group_deps*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_4_grad/b_switch*
_output_shapes

:

�
-rnn/gradients/rnn/rnn/while/Enter_2_grad/ExitExitArnn/gradients/rnn/rnn/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes
: 
�
-rnn/gradients/rnn/rnn/while/Enter_3_grad/ExitExitArnn/gradients/rnn/rnn/while/Merge_3_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
-rnn/gradients/rnn/rnn/while/Enter_4_grad/ExitExitArnn/gradients/rnn/rnn/while/Merge_4_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
frnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lrnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterCrnn/gradients/rnn/rnn/while/Merge_2_grad/tuple/control_dependency_1*6
_class,
*(loc:@rnn/rnn/while/basic_lstm_cell/Mul_2*
sourcernn/gradients*
_output_shapes

:: 
�
lrnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn/rnn/TensorArray*
T0*6
_class,
*(loc:@rnn/rnn/while/basic_lstm_cell/Mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
brnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityCrnn/gradients/rnn/rnn/while/Merge_2_grad/tuple/control_dependency_1g^rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*6
_class,
*(loc:@rnn/rnn/while/basic_lstm_cell/Mul_2*
_output_shapes
: 
�
Vrnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3frnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3arnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2brnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*'
_output_shapes
:���������

�
\rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*+
_class!
loc:@rnn/rnn/while/Identity_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
\rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2\rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*
	elem_type0*+
_class!
loc:@rnn/rnn/while/Identity_1*

stack_name *
_output_shapes
:
�
\rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnter\rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*+

frame_namernn/rnn/while/while_context
�
brnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2\rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterrnn/rnn/while/Identity_1^rnn/gradients/Add*
T0*
_output_shapes
: *
swap_memory( 
�
arnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2grnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^rnn/gradients/Sub*
_output_shapes
: *
	elem_type0
�
grnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnter\rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
]rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerb^rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2L^rnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2F^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2H^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2F^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2H^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2D^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2F^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2J^rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2L^rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
�
Urnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpD^rnn/gradients/rnn/rnn/while/Merge_2_grad/tuple/control_dependency_1W^rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
]rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityVrnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3V^rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*i
_class_
][loc:@rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
_output_shapes

:

�
_rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityCrnn/gradients/rnn/rnn/while/Merge_2_grad/tuple/control_dependency_1V^rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
�
rnn/gradients/AddNAddNCrnn/gradients/rnn/rnn/while/Merge_4_grad/tuple/control_dependency_1]rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_4_grad/b_switch*
N*
_output_shapes

:

�
:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/MulMulrnn/gradients/AddNErnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2*
T0*
_output_shapes

:

�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/ConstConst*:
_class0
.,loc:@rnn/rnn/while/basic_lstm_cell/Sigmoid_2*
valueB :
���������*
dtype0*
_output_shapes
: 
�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_accStackV2@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Const*
	elem_type0*:
_class0
.,loc:@rnn/rnn/while/basic_lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:
�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/EnterEnter@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*+

frame_namernn/rnn/while/while_context
�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2StackPushV2@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Enter'rnn/rnn/while/basic_lstm_cell/Sigmoid_2^rnn/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Ernn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2
StackPopV2Krnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/Enter^rnn/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Krnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/EnterEnter@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
<rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1Mulrnn/gradients/AddNGrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2*
T0*
_output_shapes

:

�
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/ConstConst*7
_class-
+)loc:@rnn/rnn/while/basic_lstm_cell/Tanh_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_accStackV2Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Const*7
_class-
+)loc:@rnn/rnn/while/basic_lstm_cell/Tanh_1*

stack_name *
_output_shapes
:*
	elem_type0
�
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/EnterEnterBrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*+

frame_namernn/rnn/while/while_context
�
Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2StackPushV2Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter$rnn/rnn/while/basic_lstm_cell/Tanh_1^rnn/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Grnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2
StackPopV2Mrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/Enter^rnn/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Mrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/EnterEnterBrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
Grnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_depsNoOp;^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul=^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1
�
Ornn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependencyIdentity:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/MulH^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*M
_classC
A?loc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul*
_output_shapes

:

�
Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1Identity<rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1H^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*O
_classE
CAloc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1*
_output_shapes

:

�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradGrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2Ornn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradErnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
9rnn/gradients/rnn/rnn/while/Switch_2_grad_1/NextIterationNextIteration_rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
rnn/gradients/AddN_1AddNCrnn/gradients/rnn/rnn/while/Merge_3_grad/tuple/control_dependency_1@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_3_grad/b_switch*
N*
_output_shapes

:

f
Grnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_depsNoOp^rnn/gradients/AddN_1
�
Ornn/gradients/rnn/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyIdentityrnn/gradients/AddN_1H^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Identityrnn/gradients/AddN_1H^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
8rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/MulMulOrnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyCrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2*
T0*
_output_shapes

:

�
>rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/ConstConst*8
_class.
,*loc:@rnn/rnn/while/basic_lstm_cell/Sigmoid*
valueB :
���������*
dtype0*
_output_shapes
: 
�
>rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_accStackV2>rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/Const*
	elem_type0*8
_class.
,*loc:@rnn/rnn/while/basic_lstm_cell/Sigmoid*

stack_name *
_output_shapes
:
�
>rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/EnterEnter>rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*+

frame_namernn/rnn/while/while_context
�
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2StackPushV2>rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/Enter%rnn/rnn/while/basic_lstm_cell/Sigmoid^rnn/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Crnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2
StackPopV2Irnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/Enter^rnn/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Irnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/EnterEnter>rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1MulOrnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyErnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2*
T0*
_output_shapes

:

�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/ConstConst*+
_class!
loc:@rnn/rnn/while/Identity_3*
valueB :
���������*
dtype0*
_output_shapes
: 
�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_accStackV2@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Const*
	elem_type0*+
_class!
loc:@rnn/rnn/while/Identity_3*

stack_name *
_output_shapes
:
�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/EnterEnter@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*+

frame_namernn/rnn/while/while_context
�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2StackPushV2@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Enterrnn/rnn/while/Identity_3^rnn/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Ernn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
StackPopV2Krnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/Enter^rnn/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Krnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/EnterEnter@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
Ernn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_depsNoOp9^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul;^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1
�
Mrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependencyIdentity8rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/MulF^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul*
_output_shapes

:

�
Ornn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1Identity:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1F^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1*
_output_shapes

:

�
:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/MulMulQrnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Ernn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2*
T0*
_output_shapes

:

�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/ConstConst*5
_class+
)'loc:@rnn/rnn/while/basic_lstm_cell/Tanh*
valueB :
���������*
dtype0*
_output_shapes
: 
�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_accStackV2@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Const*
	elem_type0*5
_class+
)'loc:@rnn/rnn/while/basic_lstm_cell/Tanh*

stack_name *
_output_shapes
:
�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/EnterEnter@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*+

frame_namernn/rnn/while/while_context
�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2StackPushV2@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Enter"rnn/rnn/while/basic_lstm_cell/Tanh^rnn/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Ernn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2
StackPopV2Krnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/Enter^rnn/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Krnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/EnterEnter@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
<rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1MulQrnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Grnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2*
T0*
_output_shapes

:

�
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/ConstConst*:
_class0
.,loc:@rnn/rnn/while/basic_lstm_cell/Sigmoid_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_accStackV2Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Const*
	elem_type0*:
_class0
.,loc:@rnn/rnn/while/basic_lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:
�
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/EnterEnterBrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*+

frame_namernn/rnn/while/while_context
�
Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2StackPushV2Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter'rnn/rnn/while/basic_lstm_cell/Sigmoid_1^rnn/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Grnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2
StackPopV2Mrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/Enter^rnn/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Mrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/EnterEnterBrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
Grnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_depsNoOp;^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul=^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1
�
Ornn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependencyIdentity:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/MulH^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul*
_output_shapes

:

�
Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1Identity<rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1H^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1*
_output_shapes

:

�
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradCrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2Ornn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradGrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2Ornn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
>rnn/gradients/rnn/rnn/while/basic_lstm_cell/Tanh_grad/TanhGradTanhGradErnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
9rnn/gradients/rnn/rnn/while/Switch_3_grad_1/NextIterationNextIterationMrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/ShapeConst^rnn/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
<rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Shape_1Const^rnn/gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
�
Jrnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgs:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Shape<rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/SumSumDrnn/gradients/rnn/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradJrnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes

:
*
	keep_dims( *

Tidx0
�
<rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/ReshapeReshape8rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Sum:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Sum_1SumDrnn/gradients/rnn/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradLrnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
>rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Reshape_1Reshape:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Sum_1<rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Ernn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/tuple/group_depsNoOp=^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Reshape?^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Reshape_1
�
Mrnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependencyIdentity<rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/ReshapeF^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*O
_classE
CAloc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Reshape*
_output_shapes

:

�
Ornn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependency_1Identity>rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Reshape_1F^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Reshape_1*
_output_shapes
: 
�
=rnn/gradients/rnn/rnn/while/basic_lstm_cell/split_grad/concatConcatV2Frnn/gradients/rnn/rnn/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGrad>rnn/gradients/rnn/rnn/while/basic_lstm_cell/Tanh_grad/TanhGradMrnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependencyFrnn/gradients/rnn/rnn/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradCrnn/gradients/rnn/rnn/while/basic_lstm_cell/split_grad/concat/Const*

Tidx0*
T0*
N*
_output_shapes

:(
�
Crnn/gradients/rnn/rnn/while/basic_lstm_cell/split_grad/concat/ConstConst^rnn/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad=rnn/gradients/rnn/rnn/while/basic_lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:(
�
Irnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOpE^rnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGrad>^rnn/gradients/rnn/rnn/while/basic_lstm_cell/split_grad/concat
�
Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity=rnn/gradients/rnn/rnn/while/basic_lstm_cell/split_grad/concatJ^rnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/split_grad/concat*
_output_shapes

:(
�
Srnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityDrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradJ^rnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes
:(
�
>rnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMulMatMulQrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyDrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(*
T0*
_output_shapes

:*
transpose_a( 
�
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter#rnn/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1MatMulKrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:(*
transpose_a(*
transpose_b( 
�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/ConstConst*7
_class-
+)loc:@rnn/rnn/while/basic_lstm_cell/concat*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Frnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Const*
	elem_type0*7
_class-
+)loc:@rnn/rnn/while/basic_lstm_cell/concat*

stack_name *
_output_shapes
:
�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnterFrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*+

frame_namernn/rnn/while/while_context
�
Lrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Frnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter$rnn/rnn/while/basic_lstm_cell/concat^rnn/gradients/Add*
T0*
_output_shapes

:*
swap_memory( 
�
Krnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^rnn/gradients/Sub*
_output_shapes

:*
	elem_type0
�
Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterFrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOp?^rnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMulA^rnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1
�
Prnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentity>rnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMulI^rnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul*
_output_shapes

:
�
Rrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1Identity@rnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1I^rnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1*
_output_shapes

:(
�
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB(*    *
dtype0*
_output_shapes
:(
�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterDrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:(*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeFrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Lrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes

:(: 
�
Ernn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchFrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2rnn/gradients/b_count_2*
T0* 
_output_shapes
:(:(
�
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/AddAddGrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1Srnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:(
�
Lrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationBrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes
:(
�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitErnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes
:(
�
=rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ConstConst^rnn/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
<rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/RankConst^rnn/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
;rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/modFloorMod=rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/Const<rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
�
=rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeShapernn/rnn/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
�
>rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeNShapeNIrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2Krnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1*
T0*
out_type0*
N* 
_output_shapes
::
�
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/ConstConst*2
_class(
&$loc:@rnn/rnn/while/TensorArrayReadV3*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_accStackV2Drnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const*
	elem_type0*2
_class(
&$loc:@rnn/rnn/while/TensorArrayReadV3*

stack_name *
_output_shapes
:
�
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/EnterEnterDrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*+

frame_namernn/rnn/while/while_context
�
Jrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Drnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enterrnn/rnn/while/TensorArrayReadV3^rnn/gradients/Add*
T0*'
_output_shapes
:���������
*
swap_memory( 
�
Irnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2Ornn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^rnn/gradients/Sub*'
_output_shapes
:���������
*
	elem_type0
�
Ornn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterDrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const_1Const*+
_class!
loc:@rnn/rnn/while/Identity_4*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1StackV2Frnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const_1*
	elem_type0*+
_class!
loc:@rnn/rnn/while/Identity_4*

stack_name *
_output_shapes
:
�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1EnterFrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*+

frame_namernn/rnn/while/while_context
�
Lrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1StackPushV2Frnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1rnn/rnn/while/Identity_4^rnn/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Krnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
StackPopV2Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/Enter^rnn/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/EnterEnterFrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffset;rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/mod>rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN@rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
�
=rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/SliceSlicePrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyDrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ConcatOffset>rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN*
Index0*
T0*'
_output_shapes
:���������

�
?rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/Slice_1SlicePrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyFrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ConcatOffset:1@rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*
_output_shapes

:

�
Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/tuple/group_depsNoOp>^rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/Slice@^rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/Slice_1
�
Prnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependencyIdentity=rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/SliceI^rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*P
_classF
DBloc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/Slice*'
_output_shapes
:���������

�
Rrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1Identity?rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/Slice_1I^rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*R
_classH
FDloc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes

:

�
Crnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB(*    *
dtype0*
_output_shapes

:(
�
Ernn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1EnterCrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:(*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
Ernn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2MergeErnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Krnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N* 
_output_shapes
:(: 
�
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitchErnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2rnn/gradients/b_count_2*
T0*(
_output_shapes
:(:(
�
Arnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul/Enter_grad/AddAddFrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Switch:1Rrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:(
�
Krnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationArnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Add*
T0*
_output_shapes

:(
�
Ernn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3ExitDrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Switch*
T0*
_output_shapes

:(
�
Trnn/gradients/rnn/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Zrnn/gradients/rnn/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter\rnn/gradients/rnn/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^rnn/gradients/Sub*8
_class.
,*loc:@rnn/rnn/while/TensorArrayReadV3/Enter*
sourcernn/gradients*
_output_shapes

:: 
�
Zrnn/gradients/rnn/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn/rnn/TensorArray_1*
T0*8
_class.
,*loc:@rnn/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
:*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
\rnn/gradients/rnn/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterBrnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*8
_class.
,*loc:@rnn/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
: *9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
Prnn/gradients/rnn/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentity\rnn/gradients/rnn/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1U^rnn/gradients/rnn/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*8
_class.
,*loc:@rnn/rnn/while/TensorArrayReadV3/Enter*
_output_shapes
: 
�
Vrnn/gradients/rnn/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Trnn/gradients/rnn/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3arnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Prnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependencyPrnn/gradients/rnn/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
�
@rnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Brnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Enter@rnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
Brnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeBrnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Hrnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
�
Arnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchBrnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2rnn/gradients/b_count_2*
T0*
_output_shapes
: : 
�
>rnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddCrnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Vrnn/gradients/rnn/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
Hrnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration>rnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
�
Brnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitArnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 
�
9rnn/gradients/rnn/rnn/while/Switch_4_grad_1/NextIterationNextIterationRrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
wrnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn/rnn/TensorArray_1Brnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*(
_class
loc:@rnn/rnn/TensorArray_1*
sourcernn/gradients*
_output_shapes

:: 
�
srnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityBrnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3x^rnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*(
_class
loc:@rnn/rnn/TensorArray_1*
_output_shapes
: 
�
irnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3wrnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3 rnn/rnn/TensorArrayUnstack/rangesrnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*4
_output_shapes"
 :������������������
*
element_shape:
�
frnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpj^rnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3C^rnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
�
nrnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityirnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3g^rnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*|
_classr
pnloc:@rnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3*+
_output_shapes
:���������

�
prnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityBrnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3g^rnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*U
_classK
IGloc:@rnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes
: 
�
6rnn/gradients/rnn/rnn/transpose_grad/InvertPermutationInvertPermutationrnn/rnn/concat*
T0*
_output_shapes
:
�
.rnn/gradients/rnn/rnn/transpose_grad/transpose	Transposenrnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency6rnn/gradients/rnn/rnn/transpose_grad/InvertPermutation*
Tperm0*
T0*+
_output_shapes
:���������

m
&rnn/gradients/rnn/Reshape_1_grad/ShapeShapernn/add*
T0*
out_type0*
_output_shapes
:
�
(rnn/gradients/rnn/Reshape_1_grad/ReshapeReshape.rnn/gradients/rnn/rnn/transpose_grad/transpose&rnn/gradients/rnn/Reshape_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

j
 rnn/gradients/rnn/add_grad/ShapeShape
rnn/MatMul*
T0*
out_type0*
_output_shapes
:
l
"rnn/gradients/rnn/add_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
�
0rnn/gradients/rnn/add_grad/BroadcastGradientArgsBroadcastGradientArgs rnn/gradients/rnn/add_grad/Shape"rnn/gradients/rnn/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
rnn/gradients/rnn/add_grad/SumSum(rnn/gradients/rnn/Reshape_1_grad/Reshape0rnn/gradients/rnn/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
"rnn/gradients/rnn/add_grad/ReshapeReshapernn/gradients/rnn/add_grad/Sum rnn/gradients/rnn/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
 rnn/gradients/rnn/add_grad/Sum_1Sum(rnn/gradients/rnn/Reshape_1_grad/Reshape2rnn/gradients/rnn/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
$rnn/gradients/rnn/add_grad/Reshape_1Reshape rnn/gradients/rnn/add_grad/Sum_1"rnn/gradients/rnn/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:


+rnn/gradients/rnn/add_grad/tuple/group_depsNoOp#^rnn/gradients/rnn/add_grad/Reshape%^rnn/gradients/rnn/add_grad/Reshape_1
�
3rnn/gradients/rnn/add_grad/tuple/control_dependencyIdentity"rnn/gradients/rnn/add_grad/Reshape,^rnn/gradients/rnn/add_grad/tuple/group_deps*
T0*5
_class+
)'loc:@rnn/gradients/rnn/add_grad/Reshape*'
_output_shapes
:���������

�
5rnn/gradients/rnn/add_grad/tuple/control_dependency_1Identity$rnn/gradients/rnn/add_grad/Reshape_1,^rnn/gradients/rnn/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@rnn/gradients/rnn/add_grad/Reshape_1*
_output_shapes
:

�
$rnn/gradients/rnn/MatMul_grad/MatMulMatMul3rnn/gradients/rnn/add_grad/tuple/control_dependencylayer/weights/Variable/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
&rnn/gradients/rnn/MatMul_grad/MatMul_1MatMulrnn/Reshape3rnn/gradients/rnn/add_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
�
.rnn/gradients/rnn/MatMul_grad/tuple/group_depsNoOp%^rnn/gradients/rnn/MatMul_grad/MatMul'^rnn/gradients/rnn/MatMul_grad/MatMul_1
�
6rnn/gradients/rnn/MatMul_grad/tuple/control_dependencyIdentity$rnn/gradients/rnn/MatMul_grad/MatMul/^rnn/gradients/rnn/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@rnn/gradients/rnn/MatMul_grad/MatMul*'
_output_shapes
:���������
�
8rnn/gradients/rnn/MatMul_grad/tuple/control_dependency_1Identity&rnn/gradients/rnn/MatMul_grad/MatMul_1/^rnn/gradients/rnn/MatMul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@rnn/gradients/rnn/MatMul_grad/MatMul_1*
_output_shapes

:

�
rnn/beta1_power/initial_valueConst*(
_class
loc:@layer/biases/Variable*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
rnn/beta1_power
VariableV2*
shared_name *(
_class
loc:@layer/biases/Variable*
	container *
shape: *
dtype0*
_output_shapes
: 
�
rnn/beta1_power/AssignAssignrnn/beta1_powerrnn/beta1_power/initial_value*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: 
|
rnn/beta1_power/readIdentityrnn/beta1_power*
T0*(
_class
loc:@layer/biases/Variable*
_output_shapes
: 
�
rnn/beta2_power/initial_valueConst*(
_class
loc:@layer/biases/Variable*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
rnn/beta2_power
VariableV2*(
_class
loc:@layer/biases/Variable*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
rnn/beta2_power/AssignAssignrnn/beta2_powerrnn/beta2_power/initial_value*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: 
|
rnn/beta2_power/readIdentityrnn/beta2_power*
T0*(
_class
loc:@layer/biases/Variable*
_output_shapes
: 
�
1rnn/layer/weights/Variable/Adam/Initializer/zerosConst*)
_class
loc:@layer/weights/Variable*
valueB
*    *
dtype0*
_output_shapes

:

�
rnn/layer/weights/Variable/Adam
VariableV2*
shared_name *)
_class
loc:@layer/weights/Variable*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
&rnn/layer/weights/Variable/Adam/AssignAssignrnn/layer/weights/Variable/Adam1rnn/layer/weights/Variable/Adam/Initializer/zeros*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
$rnn/layer/weights/Variable/Adam/readIdentityrnn/layer/weights/Variable/Adam*
T0*)
_class
loc:@layer/weights/Variable*
_output_shapes

:

�
3rnn/layer/weights/Variable/Adam_1/Initializer/zerosConst*)
_class
loc:@layer/weights/Variable*
valueB
*    *
dtype0*
_output_shapes

:

�
!rnn/layer/weights/Variable/Adam_1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *)
_class
loc:@layer/weights/Variable*
	container *
shape
:

�
(rnn/layer/weights/Variable/Adam_1/AssignAssign!rnn/layer/weights/Variable/Adam_13rnn/layer/weights/Variable/Adam_1/Initializer/zeros*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
&rnn/layer/weights/Variable/Adam_1/readIdentity!rnn/layer/weights/Variable/Adam_1*
T0*)
_class
loc:@layer/weights/Variable*
_output_shapes

:

�
3rnn/layer/weights/Variable_1/Adam/Initializer/zerosConst*+
_class!
loc:@layer/weights/Variable_1*
valueB
*    *
dtype0*
_output_shapes

:

�
!rnn/layer/weights/Variable_1/Adam
VariableV2*
shared_name *+
_class!
loc:@layer/weights/Variable_1*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
(rnn/layer/weights/Variable_1/Adam/AssignAssign!rnn/layer/weights/Variable_1/Adam3rnn/layer/weights/Variable_1/Adam/Initializer/zeros*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
&rnn/layer/weights/Variable_1/Adam/readIdentity!rnn/layer/weights/Variable_1/Adam*
T0*+
_class!
loc:@layer/weights/Variable_1*
_output_shapes

:

�
5rnn/layer/weights/Variable_1/Adam_1/Initializer/zerosConst*+
_class!
loc:@layer/weights/Variable_1*
valueB
*    *
dtype0*
_output_shapes

:

�
#rnn/layer/weights/Variable_1/Adam_1
VariableV2*
shared_name *+
_class!
loc:@layer/weights/Variable_1*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
*rnn/layer/weights/Variable_1/Adam_1/AssignAssign#rnn/layer/weights/Variable_1/Adam_15rnn/layer/weights/Variable_1/Adam_1/Initializer/zeros*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
(rnn/layer/weights/Variable_1/Adam_1/readIdentity#rnn/layer/weights/Variable_1/Adam_1*
T0*+
_class!
loc:@layer/weights/Variable_1*
_output_shapes

:

�
0rnn/layer/biases/Variable/Adam/Initializer/zerosConst*(
_class
loc:@layer/biases/Variable*
valueB
*    *
dtype0*
_output_shapes
:

�
rnn/layer/biases/Variable/Adam
VariableV2*
shared_name *(
_class
loc:@layer/biases/Variable*
	container *
shape:
*
dtype0*
_output_shapes
:

�
%rnn/layer/biases/Variable/Adam/AssignAssignrnn/layer/biases/Variable/Adam0rnn/layer/biases/Variable/Adam/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

�
#rnn/layer/biases/Variable/Adam/readIdentityrnn/layer/biases/Variable/Adam*
T0*(
_class
loc:@layer/biases/Variable*
_output_shapes
:

�
2rnn/layer/biases/Variable/Adam_1/Initializer/zerosConst*(
_class
loc:@layer/biases/Variable*
valueB
*    *
dtype0*
_output_shapes
:

�
 rnn/layer/biases/Variable/Adam_1
VariableV2*
shared_name *(
_class
loc:@layer/biases/Variable*
	container *
shape:
*
dtype0*
_output_shapes
:

�
'rnn/layer/biases/Variable/Adam_1/AssignAssign rnn/layer/biases/Variable/Adam_12rnn/layer/biases/Variable/Adam_1/Initializer/zeros*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
%rnn/layer/biases/Variable/Adam_1/readIdentity rnn/layer/biases/Variable/Adam_1*
T0*(
_class
loc:@layer/biases/Variable*
_output_shapes
:

�
2rnn/layer/biases/Variable_1/Adam/Initializer/zerosConst**
_class 
loc:@layer/biases/Variable_1*
valueB*    *
dtype0*
_output_shapes
:
�
 rnn/layer/biases/Variable_1/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@layer/biases/Variable_1*
	container 
�
'rnn/layer/biases/Variable_1/Adam/AssignAssign rnn/layer/biases/Variable_1/Adam2rnn/layer/biases/Variable_1/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
%rnn/layer/biases/Variable_1/Adam/readIdentity rnn/layer/biases/Variable_1/Adam*
T0**
_class 
loc:@layer/biases/Variable_1*
_output_shapes
:
�
4rnn/layer/biases/Variable_1/Adam_1/Initializer/zerosConst**
_class 
loc:@layer/biases/Variable_1*
valueB*    *
dtype0*
_output_shapes
:
�
"rnn/layer/biases/Variable_1/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@layer/biases/Variable_1*
	container 
�
)rnn/layer/biases/Variable_1/Adam_1/AssignAssign"rnn/layer/biases/Variable_1/Adam_14rnn/layer/biases/Variable_1/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
'rnn/layer/biases/Variable_1/Adam_1/readIdentity"rnn/layer/biases/Variable_1/Adam_1*
T0**
_class 
loc:@layer/biases/Variable_1*
_output_shapes
:
�
9rnn/rnn/rnn/basic_lstm_cell/kernel/Adam/Initializer/zerosConst*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
valueB(*    *
dtype0*
_output_shapes

:(
�
'rnn/rnn/rnn/basic_lstm_cell/kernel/Adam
VariableV2*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
	container *
shape
:(*
dtype0*
_output_shapes

:(*
shared_name 
�
.rnn/rnn/rnn/basic_lstm_cell/kernel/Adam/AssignAssign'rnn/rnn/rnn/basic_lstm_cell/kernel/Adam9rnn/rnn/rnn/basic_lstm_cell/kernel/Adam/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
,rnn/rnn/rnn/basic_lstm_cell/kernel/Adam/readIdentity'rnn/rnn/rnn/basic_lstm_cell/kernel/Adam*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
;rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zerosConst*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
valueB(*    *
dtype0*
_output_shapes

:(
�
)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:(*
shared_name *1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
	container *
shape
:(
�
0rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1/AssignAssign)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1;rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
.rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1/readIdentity)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
_output_shapes

:(
�
7rnn/rnn/rnn/basic_lstm_cell/bias/Adam/Initializer/zerosConst*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
valueB(*    *
dtype0*
_output_shapes
:(
�
%rnn/rnn/rnn/basic_lstm_cell/bias/Adam
VariableV2*
dtype0*
_output_shapes
:(*
shared_name */
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
	container *
shape:(
�
,rnn/rnn/rnn/basic_lstm_cell/bias/Adam/AssignAssign%rnn/rnn/rnn/basic_lstm_cell/bias/Adam7rnn/rnn/rnn/basic_lstm_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
*rnn/rnn/rnn/basic_lstm_cell/bias/Adam/readIdentity%rnn/rnn/rnn/basic_lstm_cell/bias/Adam*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
_output_shapes
:(
�
9rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
valueB(*    *
dtype0*
_output_shapes
:(
�
'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:(*
shared_name */
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
	container *
shape:(
�
.rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1/AssignAssign'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_19rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
,rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1/readIdentity'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
_output_shapes
:(
[
rnn/Adam/learning_rateConst*
valueB
 *RI9*
dtype0*
_output_shapes
: 
S
rnn/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
S
rnn/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
U
rnn/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
0rnn/Adam/update_layer/weights/Variable/ApplyAdam	ApplyAdamlayer/weights/Variablernn/layer/weights/Variable/Adam!rnn/layer/weights/Variable/Adam_1rnn/beta1_power/readrnn/beta2_power/readrnn/Adam/learning_raternn/Adam/beta1rnn/Adam/beta2rnn/Adam/epsilon8rnn/gradients/rnn/MatMul_grad/tuple/control_dependency_1*
T0*)
_class
loc:@layer/weights/Variable*
use_nesterov( *
_output_shapes

:
*
use_locking( 
�
2rnn/Adam/update_layer/weights/Variable_1/ApplyAdam	ApplyAdamlayer/weights/Variable_1!rnn/layer/weights/Variable_1/Adam#rnn/layer/weights/Variable_1/Adam_1rnn/beta1_power/readrnn/beta2_power/readrnn/Adam/learning_raternn/Adam/beta1rnn/Adam/beta2rnn/Adam/epsilon:rnn/gradients/rnn/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@layer/weights/Variable_1*
use_nesterov( *
_output_shapes

:

�
/rnn/Adam/update_layer/biases/Variable/ApplyAdam	ApplyAdamlayer/biases/Variablernn/layer/biases/Variable/Adam rnn/layer/biases/Variable/Adam_1rnn/beta1_power/readrnn/beta2_power/readrnn/Adam/learning_raternn/Adam/beta1rnn/Adam/beta2rnn/Adam/epsilon5rnn/gradients/rnn/add_grad/tuple/control_dependency_1*
use_locking( *
T0*(
_class
loc:@layer/biases/Variable*
use_nesterov( *
_output_shapes
:

�
1rnn/Adam/update_layer/biases/Variable_1/ApplyAdam	ApplyAdamlayer/biases/Variable_1 rnn/layer/biases/Variable_1/Adam"rnn/layer/biases/Variable_1/Adam_1rnn/beta1_power/readrnn/beta2_power/readrnn/Adam/learning_raternn/Adam/beta1rnn/Adam/beta2rnn/Adam/epsilon7rnn/gradients/rnn/preds_grad/tuple/control_dependency_1*
T0**
_class 
loc:@layer/biases/Variable_1*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
8rnn/Adam/update_rnn/rnn/basic_lstm_cell/kernel/ApplyAdam	ApplyAdamrnn/rnn/basic_lstm_cell/kernel'rnn/rnn/rnn/basic_lstm_cell/kernel/Adam)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn/beta1_power/readrnn/beta2_power/readrnn/Adam/learning_raternn/Adam/beta1rnn/Adam/beta2rnn/Adam/epsilonErnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
use_locking( *
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
use_nesterov( *
_output_shapes

:(
�
6rnn/Adam/update_rnn/rnn/basic_lstm_cell/bias/ApplyAdam	ApplyAdamrnn/rnn/basic_lstm_cell/bias%rnn/rnn/rnn/basic_lstm_cell/bias/Adam'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1rnn/beta1_power/readrnn/beta2_power/readrnn/Adam/learning_raternn/Adam/beta1rnn/Adam/beta2rnn/Adam/epsilonFrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
use_nesterov( *
_output_shapes
:(
�
rnn/Adam/mulMulrnn/beta1_power/readrnn/Adam/beta10^rnn/Adam/update_layer/biases/Variable/ApplyAdam2^rnn/Adam/update_layer/biases/Variable_1/ApplyAdam1^rnn/Adam/update_layer/weights/Variable/ApplyAdam3^rnn/Adam/update_layer/weights/Variable_1/ApplyAdam7^rnn/Adam/update_rnn/rnn/basic_lstm_cell/bias/ApplyAdam9^rnn/Adam/update_rnn/rnn/basic_lstm_cell/kernel/ApplyAdam*
T0*(
_class
loc:@layer/biases/Variable*
_output_shapes
: 
�
rnn/Adam/AssignAssignrnn/beta1_powerrnn/Adam/mul*
use_locking( *
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn/Adam/mul_1Mulrnn/beta2_power/readrnn/Adam/beta20^rnn/Adam/update_layer/biases/Variable/ApplyAdam2^rnn/Adam/update_layer/biases/Variable_1/ApplyAdam1^rnn/Adam/update_layer/weights/Variable/ApplyAdam3^rnn/Adam/update_layer/weights/Variable_1/ApplyAdam7^rnn/Adam/update_rnn/rnn/basic_lstm_cell/bias/ApplyAdam9^rnn/Adam/update_rnn/rnn/basic_lstm_cell/kernel/ApplyAdam*
T0*(
_class
loc:@layer/biases/Variable*
_output_shapes
: 
�
rnn/Adam/Assign_1Assignrnn/beta2_powerrnn/Adam/mul_1*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking( 
�
rnn/AdamNoOp^rnn/Adam/Assign^rnn/Adam/Assign_10^rnn/Adam/update_layer/biases/Variable/ApplyAdam2^rnn/Adam/update_layer/biases/Variable_1/ApplyAdam1^rnn/Adam/update_layer/weights/Variable/ApplyAdam3^rnn/Adam/update_layer/weights/Variable_1/ApplyAdam7^rnn/Adam/update_rnn/rnn/basic_lstm_cell/bias/ApplyAdam9^rnn/Adam/update_rnn/rnn/basic_lstm_cell/kernel/ApplyAdam
T
rnn/save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
rnn/save/SaveV2/tensor_namesConst*�
value�B�Blayer/biases/VariableBlayer/biases/Variable_1Blayer/weights/VariableBlayer/weights/Variable_1Brnn/beta1_powerBrnn/beta2_powerBrnn/layer/biases/Variable/AdamB rnn/layer/biases/Variable/Adam_1B rnn/layer/biases/Variable_1/AdamB"rnn/layer/biases/Variable_1/Adam_1Brnn/layer/weights/Variable/AdamB!rnn/layer/weights/Variable/Adam_1B!rnn/layer/weights/Variable_1/AdamB#rnn/layer/weights/Variable_1/Adam_1Brnn/rnn/basic_lstm_cell/biasBrnn/rnn/basic_lstm_cell/kernelB%rnn/rnn/rnn/basic_lstm_cell/bias/AdamB'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1B'rnn/rnn/rnn/basic_lstm_cell/kernel/AdamB)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1*
dtype0*
_output_shapes
:
�
 rnn/save/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
rnn/save/SaveV2SaveV2rnn/save/Constrnn/save/SaveV2/tensor_names rnn/save/SaveV2/shape_and_sliceslayer/biases/Variablelayer/biases/Variable_1layer/weights/Variablelayer/weights/Variable_1rnn/beta1_powerrnn/beta2_powerrnn/layer/biases/Variable/Adam rnn/layer/biases/Variable/Adam_1 rnn/layer/biases/Variable_1/Adam"rnn/layer/biases/Variable_1/Adam_1rnn/layer/weights/Variable/Adam!rnn/layer/weights/Variable/Adam_1!rnn/layer/weights/Variable_1/Adam#rnn/layer/weights/Variable_1/Adam_1rnn/rnn/basic_lstm_cell/biasrnn/rnn/basic_lstm_cell/kernel%rnn/rnn/rnn/basic_lstm_cell/bias/Adam'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1'rnn/rnn/rnn/basic_lstm_cell/kernel/Adam)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1*"
dtypes
2
�
rnn/save/control_dependencyIdentityrnn/save/Const^rnn/save/SaveV2*
T0*!
_class
loc:@rnn/save/Const*
_output_shapes
: 
�
rnn/save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�Blayer/biases/VariableBlayer/biases/Variable_1Blayer/weights/VariableBlayer/weights/Variable_1Brnn/beta1_powerBrnn/beta2_powerBrnn/layer/biases/Variable/AdamB rnn/layer/biases/Variable/Adam_1B rnn/layer/biases/Variable_1/AdamB"rnn/layer/biases/Variable_1/Adam_1Brnn/layer/weights/Variable/AdamB!rnn/layer/weights/Variable/Adam_1B!rnn/layer/weights/Variable_1/AdamB#rnn/layer/weights/Variable_1/Adam_1Brnn/rnn/basic_lstm_cell/biasBrnn/rnn/basic_lstm_cell/kernelB%rnn/rnn/rnn/basic_lstm_cell/bias/AdamB'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1B'rnn/rnn/rnn/basic_lstm_cell/kernel/AdamB)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1*
dtype0*
_output_shapes
:
�
#rnn/save/RestoreV2/shape_and_slicesConst"/device:CPU:0*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
rnn/save/RestoreV2	RestoreV2rnn/save/Constrnn/save/RestoreV2/tensor_names#rnn/save/RestoreV2/shape_and_slices"/device:CPU:0*"
dtypes
2*d
_output_shapesR
P::::::::::::::::::::
�
rnn/save/AssignAssignlayer/biases/Variablernn/save/RestoreV2*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn/save/Assign_1Assignlayer/biases/Variable_1rnn/save/RestoreV2:1*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn/save/Assign_2Assignlayer/weights/Variablernn/save/RestoreV2:2*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn/save/Assign_3Assignlayer/weights/Variable_1rnn/save/RestoreV2:3*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn/save/Assign_4Assignrnn/beta1_powerrnn/save/RestoreV2:4*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn/save/Assign_5Assignrnn/beta2_powerrnn/save/RestoreV2:5*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(
�
rnn/save/Assign_6Assignrnn/layer/biases/Variable/Adamrnn/save/RestoreV2:6*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn/save/Assign_7Assign rnn/layer/biases/Variable/Adam_1rnn/save/RestoreV2:7*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn/save/Assign_8Assign rnn/layer/biases/Variable_1/Adamrnn/save/RestoreV2:8*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn/save/Assign_9Assign"rnn/layer/biases/Variable_1/Adam_1rnn/save/RestoreV2:9*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn/save/Assign_10Assignrnn/layer/weights/Variable/Adamrnn/save/RestoreV2:10*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn/save/Assign_11Assign!rnn/layer/weights/Variable/Adam_1rnn/save/RestoreV2:11*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn/save/Assign_12Assign!rnn/layer/weights/Variable_1/Adamrnn/save/RestoreV2:12*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn/save/Assign_13Assign#rnn/layer/weights/Variable_1/Adam_1rnn/save/RestoreV2:13*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn/save/Assign_14Assignrnn/rnn/basic_lstm_cell/biasrnn/save/RestoreV2:14*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
rnn/save/Assign_15Assignrnn/rnn/basic_lstm_cell/kernelrnn/save/RestoreV2:15*
use_locking(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
rnn/save/Assign_16Assign%rnn/rnn/rnn/basic_lstm_cell/bias/Adamrnn/save/RestoreV2:16*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
rnn/save/Assign_17Assign'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1rnn/save/RestoreV2:17*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
rnn/save/Assign_18Assign'rnn/rnn/rnn/basic_lstm_cell/kernel/Adamrnn/save/RestoreV2:18*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
rnn/save/Assign_19Assign)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn/save/RestoreV2:19*
use_locking(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
rnn/save/restore_allNoOp^rnn/save/Assign^rnn/save/Assign_1^rnn/save/Assign_10^rnn/save/Assign_11^rnn/save/Assign_12^rnn/save/Assign_13^rnn/save/Assign_14^rnn/save/Assign_15^rnn/save/Assign_16^rnn/save/Assign_17^rnn/save/Assign_18^rnn/save/Assign_19^rnn/save/Assign_2^rnn/save/Assign_3^rnn/save/Assign_4^rnn/save/Assign_5^rnn/save/Assign_6^rnn/save/Assign_7^rnn/save/Assign_8^rnn/save/Assign_9
T
rnn/PlaceholderPlaceholder*
shape:*
dtype0*
_output_shapes
:
�
rnn/initNoOp^layer/biases/Variable/Assign^layer/biases/Variable_1/Assign^layer/weights/Variable/Assign ^layer/weights/Variable_1/Assign^rnn/beta1_power/Assign^rnn/beta2_power/Assign&^rnn/layer/biases/Variable/Adam/Assign(^rnn/layer/biases/Variable/Adam_1/Assign(^rnn/layer/biases/Variable_1/Adam/Assign*^rnn/layer/biases/Variable_1/Adam_1/Assign'^rnn/layer/weights/Variable/Adam/Assign)^rnn/layer/weights/Variable/Adam_1/Assign)^rnn/layer/weights/Variable_1/Adam/Assign+^rnn/layer/weights/Variable_1/Adam_1/Assign$^rnn/rnn/basic_lstm_cell/bias/Assign&^rnn/rnn/basic_lstm_cell/kernel/Assign-^rnn/rnn/rnn/basic_lstm_cell/bias/Adam/Assign/^rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1/Assign/^rnn/rnn/rnn/basic_lstm_cell/kernel/Adam/Assign1^rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1/Assign
V
rnn/save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
rnn/save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_f4ec007e432b496abd982001f42d5f76/part*
dtype0*
_output_shapes
: 
�
rnn/save_1/StringJoin
StringJoinrnn/save_1/Constrnn/save_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
W
rnn/save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
q
 rnn/save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
rnn/save_1/ShardedFilenameShardedFilenamernn/save_1/StringJoin rnn/save_1/ShardedFilename/shardrnn/save_1/num_shards"/device:CPU:0*
_output_shapes
: 
�
rnn/save_1/SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�Blayer/biases/VariableBlayer/biases/Variable_1Blayer/weights/VariableBlayer/weights/Variable_1Brnn/beta1_powerBrnn/beta2_powerBrnn/layer/biases/Variable/AdamB rnn/layer/biases/Variable/Adam_1B rnn/layer/biases/Variable_1/AdamB"rnn/layer/biases/Variable_1/Adam_1Brnn/layer/weights/Variable/AdamB!rnn/layer/weights/Variable/Adam_1B!rnn/layer/weights/Variable_1/AdamB#rnn/layer/weights/Variable_1/Adam_1Brnn/rnn/basic_lstm_cell/biasBrnn/rnn/basic_lstm_cell/kernelB%rnn/rnn/rnn/basic_lstm_cell/bias/AdamB'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1B'rnn/rnn/rnn/basic_lstm_cell/kernel/AdamB)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1*
dtype0*
_output_shapes
:
�
"rnn/save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
rnn/save_1/SaveV2SaveV2rnn/save_1/ShardedFilenamernn/save_1/SaveV2/tensor_names"rnn/save_1/SaveV2/shape_and_sliceslayer/biases/Variablelayer/biases/Variable_1layer/weights/Variablelayer/weights/Variable_1rnn/beta1_powerrnn/beta2_powerrnn/layer/biases/Variable/Adam rnn/layer/biases/Variable/Adam_1 rnn/layer/biases/Variable_1/Adam"rnn/layer/biases/Variable_1/Adam_1rnn/layer/weights/Variable/Adam!rnn/layer/weights/Variable/Adam_1!rnn/layer/weights/Variable_1/Adam#rnn/layer/weights/Variable_1/Adam_1rnn/rnn/basic_lstm_cell/biasrnn/rnn/basic_lstm_cell/kernel%rnn/rnn/rnn/basic_lstm_cell/bias/Adam'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1'rnn/rnn/rnn/basic_lstm_cell/kernel/Adam)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1"/device:CPU:0*"
dtypes
2
�
rnn/save_1/control_dependencyIdentityrnn/save_1/ShardedFilename^rnn/save_1/SaveV2"/device:CPU:0*
T0*-
_class#
!loc:@rnn/save_1/ShardedFilename*
_output_shapes
: 
�
1rnn/save_1/MergeV2Checkpoints/checkpoint_prefixesPackrnn/save_1/ShardedFilename^rnn/save_1/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
rnn/save_1/MergeV2CheckpointsMergeV2Checkpoints1rnn/save_1/MergeV2Checkpoints/checkpoint_prefixesrnn/save_1/Const"/device:CPU:0*
delete_old_dirs(
�
rnn/save_1/IdentityIdentityrnn/save_1/Const^rnn/save_1/MergeV2Checkpoints^rnn/save_1/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
�
!rnn/save_1/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�Blayer/biases/VariableBlayer/biases/Variable_1Blayer/weights/VariableBlayer/weights/Variable_1Brnn/beta1_powerBrnn/beta2_powerBrnn/layer/biases/Variable/AdamB rnn/layer/biases/Variable/Adam_1B rnn/layer/biases/Variable_1/AdamB"rnn/layer/biases/Variable_1/Adam_1Brnn/layer/weights/Variable/AdamB!rnn/layer/weights/Variable/Adam_1B!rnn/layer/weights/Variable_1/AdamB#rnn/layer/weights/Variable_1/Adam_1Brnn/rnn/basic_lstm_cell/biasBrnn/rnn/basic_lstm_cell/kernelB%rnn/rnn/rnn/basic_lstm_cell/bias/AdamB'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1B'rnn/rnn/rnn/basic_lstm_cell/kernel/AdamB)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1*
dtype0*
_output_shapes
:
�
%rnn/save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
rnn/save_1/RestoreV2	RestoreV2rnn/save_1/Const!rnn/save_1/RestoreV2/tensor_names%rnn/save_1/RestoreV2/shape_and_slices"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
�
rnn/save_1/AssignAssignlayer/biases/Variablernn/save_1/RestoreV2*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn/save_1/Assign_1Assignlayer/biases/Variable_1rnn/save_1/RestoreV2:1*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn/save_1/Assign_2Assignlayer/weights/Variablernn/save_1/RestoreV2:2*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn/save_1/Assign_3Assignlayer/weights/Variable_1rnn/save_1/RestoreV2:3*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn/save_1/Assign_4Assignrnn/beta1_powerrnn/save_1/RestoreV2:4*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(
�
rnn/save_1/Assign_5Assignrnn/beta2_powerrnn/save_1/RestoreV2:5*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn/save_1/Assign_6Assignrnn/layer/biases/Variable/Adamrnn/save_1/RestoreV2:6*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn/save_1/Assign_7Assign rnn/layer/biases/Variable/Adam_1rnn/save_1/RestoreV2:7*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn/save_1/Assign_8Assign rnn/layer/biases/Variable_1/Adamrnn/save_1/RestoreV2:8*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn/save_1/Assign_9Assign"rnn/layer/biases/Variable_1/Adam_1rnn/save_1/RestoreV2:9*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn/save_1/Assign_10Assignrnn/layer/weights/Variable/Adamrnn/save_1/RestoreV2:10*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn/save_1/Assign_11Assign!rnn/layer/weights/Variable/Adam_1rnn/save_1/RestoreV2:11*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn/save_1/Assign_12Assign!rnn/layer/weights/Variable_1/Adamrnn/save_1/RestoreV2:12*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn/save_1/Assign_13Assign#rnn/layer/weights/Variable_1/Adam_1rnn/save_1/RestoreV2:13*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn/save_1/Assign_14Assignrnn/rnn/basic_lstm_cell/biasrnn/save_1/RestoreV2:14*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
rnn/save_1/Assign_15Assignrnn/rnn/basic_lstm_cell/kernelrnn/save_1/RestoreV2:15*
use_locking(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
rnn/save_1/Assign_16Assign%rnn/rnn/rnn/basic_lstm_cell/bias/Adamrnn/save_1/RestoreV2:16*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
rnn/save_1/Assign_17Assign'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1rnn/save_1/RestoreV2:17*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
rnn/save_1/Assign_18Assign'rnn/rnn/rnn/basic_lstm_cell/kernel/Adamrnn/save_1/RestoreV2:18*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
rnn/save_1/Assign_19Assign)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn/save_1/RestoreV2:19*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
rnn/save_1/restore_shardNoOp^rnn/save_1/Assign^rnn/save_1/Assign_1^rnn/save_1/Assign_10^rnn/save_1/Assign_11^rnn/save_1/Assign_12^rnn/save_1/Assign_13^rnn/save_1/Assign_14^rnn/save_1/Assign_15^rnn/save_1/Assign_16^rnn/save_1/Assign_17^rnn/save_1/Assign_18^rnn/save_1/Assign_19^rnn/save_1/Assign_2^rnn/save_1/Assign_3^rnn/save_1/Assign_4^rnn/save_1/Assign_5^rnn/save_1/Assign_6^rnn/save_1/Assign_7^rnn/save_1/Assign_8^rnn/save_1/Assign_9
9
rnn/save_1/restore_allNoOp^rnn/save_1/restore_shard
d
rnn_1/Reshape/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
u
rnn_1/ReshapeReshapeinputsrnn_1/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:���������
�
rnn_1/MatMulMatMulrnn_1/Reshapelayer/weights/Variable/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
l
	rnn_1/addAddrnn_1/MatMullayer/biases/Variable/read*
T0*'
_output_shapes
:���������

j
rnn_1/Reshape_1/shapeConst*!
valueB"����   
   *
dtype0*
_output_shapes
:
�
rnn_1/Reshape_1Reshape	rnn_1/addrnn_1/Reshape_1/shape*
T0*
Tshape0*+
_output_shapes
:���������

l
"rnn_1/BasicLSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
n
$rnn_1/BasicLSTMCellZeroState/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
j
(rnn_1/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
#rnn_1/BasicLSTMCellZeroState/concatConcatV2"rnn_1/BasicLSTMCellZeroState/Const$rnn_1/BasicLSTMCellZeroState/Const_1(rnn_1/BasicLSTMCellZeroState/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
m
(rnn_1/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"rnn_1/BasicLSTMCellZeroState/zerosFill#rnn_1/BasicLSTMCellZeroState/concat(rnn_1/BasicLSTMCellZeroState/zeros/Const*
T0*

index_type0*
_output_shapes

:

n
$rnn_1/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
n
$rnn_1/BasicLSTMCellZeroState/Const_3Const*
valueB:
*
dtype0*
_output_shapes
:
n
$rnn_1/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
n
$rnn_1/BasicLSTMCellZeroState/Const_5Const*
valueB:
*
dtype0*
_output_shapes
:
l
*rnn_1/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
%rnn_1/BasicLSTMCellZeroState/concat_1ConcatV2$rnn_1/BasicLSTMCellZeroState/Const_4$rnn_1/BasicLSTMCellZeroState/Const_5*rnn_1/BasicLSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
o
*rnn_1/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
$rnn_1/BasicLSTMCellZeroState/zeros_1Fill%rnn_1/BasicLSTMCellZeroState/concat_1*rnn_1/BasicLSTMCellZeroState/zeros_1/Const*
T0*

index_type0*
_output_shapes

:

n
$rnn_1/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
n
$rnn_1/BasicLSTMCellZeroState/Const_7Const*
valueB:
*
dtype0*
_output_shapes
:
P
rnn_1/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
W
rnn_1/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
W
rnn_1/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
~
rnn_1/rnn/rangeRangernn_1/rnn/range/startrnn_1/rnn/Rankrnn_1/rnn/range/delta*
_output_shapes
:*

Tidx0
j
rnn_1/rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
W
rnn_1/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_1/rnn/concatConcatV2rnn_1/rnn/concat/values_0rnn_1/rnn/rangernn_1/rnn/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
rnn_1/rnn/transpose	Transposernn_1/Reshape_1rnn_1/rnn/concat*
Tperm0*
T0*+
_output_shapes
:���������

b
rnn_1/rnn/ShapeShapernn_1/rnn/transpose*
T0*
out_type0*
_output_shapes
:
g
rnn_1/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
rnn_1/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
i
rnn_1/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
rnn_1/rnn/strided_sliceStridedSlicernn_1/rnn/Shapernn_1/rnn/strided_slice/stackrnn_1/rnn/strided_slice/stack_1rnn_1/rnn/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
d
rnn_1/rnn/Shape_1Shapernn_1/rnn/transpose*
T0*
out_type0*
_output_shapes
:
i
rnn_1/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
k
!rnn_1/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
k
!rnn_1/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
rnn_1/rnn/strided_slice_1StridedSlicernn_1/rnn/Shape_1rnn_1/rnn/strided_slice_1/stack!rnn_1/rnn/strided_slice_1/stack_1!rnn_1/rnn/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
d
rnn_1/rnn/Shape_2Shapernn_1/rnn/transpose*
T0*
out_type0*
_output_shapes
:
i
rnn_1/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
k
!rnn_1/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
k
!rnn_1/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
rnn_1/rnn/strided_slice_2StridedSlicernn_1/rnn/Shape_2rnn_1/rnn/strided_slice_2/stack!rnn_1/rnn/strided_slice_2/stack_1!rnn_1/rnn/strided_slice_2/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
Z
rnn_1/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_1/rnn/ExpandDims
ExpandDimsrnn_1/rnn/strided_slice_2rnn_1/rnn/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
Y
rnn_1/rnn/ConstConst*
valueB:
*
dtype0*
_output_shapes
:
Y
rnn_1/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_1/rnn/concat_1ConcatV2rnn_1/rnn/ExpandDimsrnn_1/rnn/Constrnn_1/rnn/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
Z
rnn_1/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
rnn_1/rnn/zerosFillrnn_1/rnn/concat_1rnn_1/rnn/zeros/Const*
T0*

index_type0*'
_output_shapes
:���������

P
rnn_1/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_1/rnn/TensorArrayTensorArrayV3rnn_1/rnn/strided_slice_1*$
element_shape:���������
*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*5
tensor_array_name rnn_1/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: 
�
rnn_1/rnn/TensorArray_1TensorArrayV3rnn_1/rnn/strided_slice_1*
identical_element_shapes(*4
tensor_array_namernn_1/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *$
element_shape:���������
*
dynamic_size( *
clear_after_read(
u
"rnn_1/rnn/TensorArrayUnstack/ShapeShapernn_1/rnn/transpose*
T0*
out_type0*
_output_shapes
:
z
0rnn_1/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2rnn_1/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2rnn_1/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
*rnn_1/rnn/TensorArrayUnstack/strided_sliceStridedSlice"rnn_1/rnn/TensorArrayUnstack/Shape0rnn_1/rnn/TensorArrayUnstack/strided_slice/stack2rnn_1/rnn/TensorArrayUnstack/strided_slice/stack_12rnn_1/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
j
(rnn_1/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
j
(rnn_1/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
"rnn_1/rnn/TensorArrayUnstack/rangeRange(rnn_1/rnn/TensorArrayUnstack/range/start*rnn_1/rnn/TensorArrayUnstack/strided_slice(rnn_1/rnn/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:���������
�
Drnn_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn_1/rnn/TensorArray_1"rnn_1/rnn/TensorArrayUnstack/rangernn_1/rnn/transposernn_1/rnn/TensorArray_1:1*
T0*&
_class
loc:@rnn_1/rnn/transpose*
_output_shapes
: 
U
rnn_1/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
m
rnn_1/rnn/MaximumMaximumrnn_1/rnn/Maximum/xrnn_1/rnn/strided_slice_1*
T0*
_output_shapes
: 
k
rnn_1/rnn/MinimumMinimumrnn_1/rnn/strided_slice_1rnn_1/rnn/Maximum*
T0*
_output_shapes
: 
c
!rnn_1/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_1/rnn/while/EnterEnter!rnn_1/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *-

frame_namernn_1/rnn/while/while_context
�
rnn_1/rnn/while/Enter_1Enterrnn_1/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *-

frame_namernn_1/rnn/while/while_context
�
rnn_1/rnn/while/Enter_2Enterrnn_1/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *-

frame_namernn_1/rnn/while/while_context
�
rnn_1/rnn/while/Enter_3Enter"rnn_1/BasicLSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*-

frame_namernn_1/rnn/while/while_context
�
rnn_1/rnn/while/Enter_4Enter$rnn_1/BasicLSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*-

frame_namernn_1/rnn/while/while_context
�
rnn_1/rnn/while/MergeMergernn_1/rnn/while/Enterrnn_1/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 
�
rnn_1/rnn/while/Merge_1Mergernn_1/rnn/while/Enter_1rnn_1/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
rnn_1/rnn/while/Merge_2Mergernn_1/rnn/while/Enter_2rnn_1/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
�
rnn_1/rnn/while/Merge_3Mergernn_1/rnn/while/Enter_3rnn_1/rnn/while/NextIteration_3*
T0*
N* 
_output_shapes
:
: 
�
rnn_1/rnn/while/Merge_4Mergernn_1/rnn/while/Enter_4rnn_1/rnn/while/NextIteration_4*
T0*
N* 
_output_shapes
:
: 
p
rnn_1/rnn/while/LessLessrnn_1/rnn/while/Mergernn_1/rnn/while/Less/Enter*
T0*
_output_shapes
: 
�
rnn_1/rnn/while/Less/EnterEnterrnn_1/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *-

frame_namernn_1/rnn/while/while_context
v
rnn_1/rnn/while/Less_1Lessrnn_1/rnn/while/Merge_1rnn_1/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
�
rnn_1/rnn/while/Less_1/EnterEnterrnn_1/rnn/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *-

frame_namernn_1/rnn/while/while_context
n
rnn_1/rnn/while/LogicalAnd
LogicalAndrnn_1/rnn/while/Lessrnn_1/rnn/while/Less_1*
_output_shapes
: 
X
rnn_1/rnn/while/LoopCondLoopCondrnn_1/rnn/while/LogicalAnd*
_output_shapes
: 
�
rnn_1/rnn/while/SwitchSwitchrnn_1/rnn/while/Mergernn_1/rnn/while/LoopCond*
T0*(
_class
loc:@rnn_1/rnn/while/Merge*
_output_shapes
: : 
�
rnn_1/rnn/while/Switch_1Switchrnn_1/rnn/while/Merge_1rnn_1/rnn/while/LoopCond*
T0**
_class 
loc:@rnn_1/rnn/while/Merge_1*
_output_shapes
: : 
�
rnn_1/rnn/while/Switch_2Switchrnn_1/rnn/while/Merge_2rnn_1/rnn/while/LoopCond*
T0**
_class 
loc:@rnn_1/rnn/while/Merge_2*
_output_shapes
: : 
�
rnn_1/rnn/while/Switch_3Switchrnn_1/rnn/while/Merge_3rnn_1/rnn/while/LoopCond*
T0**
_class 
loc:@rnn_1/rnn/while/Merge_3*(
_output_shapes
:
:

�
rnn_1/rnn/while/Switch_4Switchrnn_1/rnn/while/Merge_4rnn_1/rnn/while/LoopCond*
T0**
_class 
loc:@rnn_1/rnn/while/Merge_4*(
_output_shapes
:
:

_
rnn_1/rnn/while/IdentityIdentityrnn_1/rnn/while/Switch:1*
T0*
_output_shapes
: 
c
rnn_1/rnn/while/Identity_1Identityrnn_1/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
c
rnn_1/rnn/while/Identity_2Identityrnn_1/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
k
rnn_1/rnn/while/Identity_3Identityrnn_1/rnn/while/Switch_3:1*
T0*
_output_shapes

:

k
rnn_1/rnn/while/Identity_4Identityrnn_1/rnn/while/Switch_4:1*
T0*
_output_shapes

:

r
rnn_1/rnn/while/add/yConst^rnn_1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
l
rnn_1/rnn/while/addAddrnn_1/rnn/while/Identityrnn_1/rnn/while/add/y*
T0*
_output_shapes
: 
�
!rnn_1/rnn/while/TensorArrayReadV3TensorArrayReadV3'rnn_1/rnn/while/TensorArrayReadV3/Enterrnn_1/rnn/while/Identity_1)rnn_1/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:���������

�
'rnn_1/rnn/while/TensorArrayReadV3/EnterEnterrnn_1/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_1/rnn/while/while_context
�
)rnn_1/rnn/while/TensorArrayReadV3/Enter_1EnterDrnn_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *-

frame_namernn_1/rnn/while/while_context
�
%rnn_1/rnn/while/basic_lstm_cell/ConstConst^rnn_1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
+rnn_1/rnn/while/basic_lstm_cell/concat/axisConst^rnn_1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
&rnn_1/rnn/while/basic_lstm_cell/concatConcatV2!rnn_1/rnn/while/TensorArrayReadV3rnn_1/rnn/while/Identity_4+rnn_1/rnn/while/basic_lstm_cell/concat/axis*
T0*
N*
_output_shapes

:*

Tidx0
�
&rnn_1/rnn/while/basic_lstm_cell/MatMulMatMul&rnn_1/rnn/while/basic_lstm_cell/concat,rnn_1/rnn/while/basic_lstm_cell/MatMul/Enter*
transpose_b( *
T0*
_output_shapes

:(*
transpose_a( 
�
,rnn_1/rnn/while/basic_lstm_cell/MatMul/EnterEnter#rnn/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*-

frame_namernn_1/rnn/while/while_context
�
'rnn_1/rnn/while/basic_lstm_cell/BiasAddBiasAdd&rnn_1/rnn/while/basic_lstm_cell/MatMul-rnn_1/rnn/while/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes

:(
�
-rnn_1/rnn/while/basic_lstm_cell/BiasAdd/EnterEnter!rnn/rnn/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:(*-

frame_namernn_1/rnn/while/while_context
�
'rnn_1/rnn/while/basic_lstm_cell/Const_1Const^rnn_1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
%rnn_1/rnn/while/basic_lstm_cell/splitSplit%rnn_1/rnn/while/basic_lstm_cell/Const'rnn_1/rnn/while/basic_lstm_cell/BiasAdd*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split
�
'rnn_1/rnn/while/basic_lstm_cell/Const_2Const^rnn_1/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
#rnn_1/rnn/while/basic_lstm_cell/AddAdd'rnn_1/rnn/while/basic_lstm_cell/split:2'rnn_1/rnn/while/basic_lstm_cell/Const_2*
T0*
_output_shapes

:

�
'rnn_1/rnn/while/basic_lstm_cell/SigmoidSigmoid#rnn_1/rnn/while/basic_lstm_cell/Add*
T0*
_output_shapes

:

�
#rnn_1/rnn/while/basic_lstm_cell/MulMulrnn_1/rnn/while/Identity_3'rnn_1/rnn/while/basic_lstm_cell/Sigmoid*
T0*
_output_shapes

:

�
)rnn_1/rnn/while/basic_lstm_cell/Sigmoid_1Sigmoid%rnn_1/rnn/while/basic_lstm_cell/split*
T0*
_output_shapes

:

~
$rnn_1/rnn/while/basic_lstm_cell/TanhTanh'rnn_1/rnn/while/basic_lstm_cell/split:1*
T0*
_output_shapes

:

�
%rnn_1/rnn/while/basic_lstm_cell/Mul_1Mul)rnn_1/rnn/while/basic_lstm_cell/Sigmoid_1$rnn_1/rnn/while/basic_lstm_cell/Tanh*
T0*
_output_shapes

:

�
%rnn_1/rnn/while/basic_lstm_cell/Add_1Add#rnn_1/rnn/while/basic_lstm_cell/Mul%rnn_1/rnn/while/basic_lstm_cell/Mul_1*
T0*
_output_shapes

:

~
&rnn_1/rnn/while/basic_lstm_cell/Tanh_1Tanh%rnn_1/rnn/while/basic_lstm_cell/Add_1*
T0*
_output_shapes

:

�
)rnn_1/rnn/while/basic_lstm_cell/Sigmoid_2Sigmoid'rnn_1/rnn/while/basic_lstm_cell/split:3*
T0*
_output_shapes

:

�
%rnn_1/rnn/while/basic_lstm_cell/Mul_2Mul&rnn_1/rnn/while/basic_lstm_cell/Tanh_1)rnn_1/rnn/while/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes

:

�
3rnn_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV39rnn_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn_1/rnn/while/Identity_1%rnn_1/rnn/while/basic_lstm_cell/Mul_2rnn_1/rnn/while/Identity_2*
T0*8
_class.
,*loc:@rnn_1/rnn/while/basic_lstm_cell/Mul_2*
_output_shapes
: 
�
9rnn_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn_1/rnn/TensorArray*
is_constant(*
_output_shapes
:*-

frame_namernn_1/rnn/while/while_context*
T0*8
_class.
,*loc:@rnn_1/rnn/while/basic_lstm_cell/Mul_2*
parallel_iterations 
t
rnn_1/rnn/while/add_1/yConst^rnn_1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
r
rnn_1/rnn/while/add_1Addrnn_1/rnn/while/Identity_1rnn_1/rnn/while/add_1/y*
T0*
_output_shapes
: 
d
rnn_1/rnn/while/NextIterationNextIterationrnn_1/rnn/while/add*
T0*
_output_shapes
: 
h
rnn_1/rnn/while/NextIteration_1NextIterationrnn_1/rnn/while/add_1*
T0*
_output_shapes
: 
�
rnn_1/rnn/while/NextIteration_2NextIteration3rnn_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
rnn_1/rnn/while/NextIteration_3NextIteration%rnn_1/rnn/while/basic_lstm_cell/Add_1*
T0*
_output_shapes

:

�
rnn_1/rnn/while/NextIteration_4NextIteration%rnn_1/rnn/while/basic_lstm_cell/Mul_2*
T0*
_output_shapes

:

U
rnn_1/rnn/while/ExitExitrnn_1/rnn/while/Switch*
T0*
_output_shapes
: 
Y
rnn_1/rnn/while/Exit_1Exitrnn_1/rnn/while/Switch_1*
T0*
_output_shapes
: 
Y
rnn_1/rnn/while/Exit_2Exitrnn_1/rnn/while/Switch_2*
T0*
_output_shapes
: 
a
rnn_1/rnn/while/Exit_3Exitrnn_1/rnn/while/Switch_3*
T0*
_output_shapes

:

a
rnn_1/rnn/while/Exit_4Exitrnn_1/rnn/while/Switch_4*
T0*
_output_shapes

:

�
,rnn_1/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn_1/rnn/TensorArrayrnn_1/rnn/while/Exit_2*(
_class
loc:@rnn_1/rnn/TensorArray*
_output_shapes
: 
�
&rnn_1/rnn/TensorArrayStack/range/startConst*(
_class
loc:@rnn_1/rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 
�
&rnn_1/rnn/TensorArrayStack/range/deltaConst*(
_class
loc:@rnn_1/rnn/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
�
 rnn_1/rnn/TensorArrayStack/rangeRange&rnn_1/rnn/TensorArrayStack/range/start,rnn_1/rnn/TensorArrayStack/TensorArraySizeV3&rnn_1/rnn/TensorArrayStack/range/delta*

Tidx0*(
_class
loc:@rnn_1/rnn/TensorArray*#
_output_shapes
:���������
�
.rnn_1/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn_1/rnn/TensorArray rnn_1/rnn/TensorArrayStack/rangernn_1/rnn/while/Exit_2*
element_shape
:
*(
_class
loc:@rnn_1/rnn/TensorArray*
dtype0*"
_output_shapes
:

[
rnn_1/rnn/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
R
rnn_1/rnn/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
Y
rnn_1/rnn/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
Y
rnn_1/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
rnn_1/rnn/range_1Rangernn_1/rnn/range_1/startrnn_1/rnn/Rank_1rnn_1/rnn/range_1/delta*
_output_shapes
:*

Tidx0
l
rnn_1/rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
Y
rnn_1/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_1/rnn/concat_2ConcatV2rnn_1/rnn/concat_2/values_0rnn_1/rnn/range_1rnn_1/rnn/concat_2/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
rnn_1/rnn/transpose_1	Transpose.rnn_1/rnn/TensorArrayStack/TensorArrayGatherV3rnn_1/rnn/concat_2*
Tperm0*
T0*"
_output_shapes
:

f
rnn_1/Reshape_2/shapeConst*
valueB"����
   *
dtype0*
_output_shapes
:

rnn_1/Reshape_2Reshapernn_1/rnn/transpose_1rnn_1/Reshape_2/shape*
T0*
Tshape0*
_output_shapes

:

�
rnn_1/MatMul_1MatMulrnn_1/Reshape_2layer/weights/Variable_1/read*
transpose_b( *
T0*
_output_shapes

:*
transpose_a( 
i
rnn_1/predsAddrnn_1/MatMul_1layer/biases/Variable_1/read*
T0*
_output_shapes

:
V
rnn_1/save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
rnn_1/save/SaveV2/tensor_namesConst*�
value�B�Blayer/biases/VariableBlayer/biases/Variable_1Blayer/weights/VariableBlayer/weights/Variable_1Brnn/beta1_powerBrnn/beta2_powerBrnn/layer/biases/Variable/AdamB rnn/layer/biases/Variable/Adam_1B rnn/layer/biases/Variable_1/AdamB"rnn/layer/biases/Variable_1/Adam_1Brnn/layer/weights/Variable/AdamB!rnn/layer/weights/Variable/Adam_1B!rnn/layer/weights/Variable_1/AdamB#rnn/layer/weights/Variable_1/Adam_1Brnn/rnn/basic_lstm_cell/biasBrnn/rnn/basic_lstm_cell/kernelB%rnn/rnn/rnn/basic_lstm_cell/bias/AdamB'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1B'rnn/rnn/rnn/basic_lstm_cell/kernel/AdamB)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1*
dtype0*
_output_shapes
:
�
"rnn_1/save/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
rnn_1/save/SaveV2SaveV2rnn_1/save/Constrnn_1/save/SaveV2/tensor_names"rnn_1/save/SaveV2/shape_and_sliceslayer/biases/Variablelayer/biases/Variable_1layer/weights/Variablelayer/weights/Variable_1rnn/beta1_powerrnn/beta2_powerrnn/layer/biases/Variable/Adam rnn/layer/biases/Variable/Adam_1 rnn/layer/biases/Variable_1/Adam"rnn/layer/biases/Variable_1/Adam_1rnn/layer/weights/Variable/Adam!rnn/layer/weights/Variable/Adam_1!rnn/layer/weights/Variable_1/Adam#rnn/layer/weights/Variable_1/Adam_1rnn/rnn/basic_lstm_cell/biasrnn/rnn/basic_lstm_cell/kernel%rnn/rnn/rnn/basic_lstm_cell/bias/Adam'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1'rnn/rnn/rnn/basic_lstm_cell/kernel/Adam)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1*"
dtypes
2
�
rnn_1/save/control_dependencyIdentityrnn_1/save/Const^rnn_1/save/SaveV2*
T0*#
_class
loc:@rnn_1/save/Const*
_output_shapes
: 
�
!rnn_1/save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�Blayer/biases/VariableBlayer/biases/Variable_1Blayer/weights/VariableBlayer/weights/Variable_1Brnn/beta1_powerBrnn/beta2_powerBrnn/layer/biases/Variable/AdamB rnn/layer/biases/Variable/Adam_1B rnn/layer/biases/Variable_1/AdamB"rnn/layer/biases/Variable_1/Adam_1Brnn/layer/weights/Variable/AdamB!rnn/layer/weights/Variable/Adam_1B!rnn/layer/weights/Variable_1/AdamB#rnn/layer/weights/Variable_1/Adam_1Brnn/rnn/basic_lstm_cell/biasBrnn/rnn/basic_lstm_cell/kernelB%rnn/rnn/rnn/basic_lstm_cell/bias/AdamB'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1B'rnn/rnn/rnn/basic_lstm_cell/kernel/AdamB)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1*
dtype0*
_output_shapes
:
�
%rnn_1/save/RestoreV2/shape_and_slicesConst"/device:CPU:0*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
rnn_1/save/RestoreV2	RestoreV2rnn_1/save/Const!rnn_1/save/RestoreV2/tensor_names%rnn_1/save/RestoreV2/shape_and_slices"/device:CPU:0*"
dtypes
2*d
_output_shapesR
P::::::::::::::::::::
�
rnn_1/save/AssignAssignlayer/biases/Variablernn_1/save/RestoreV2*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn_1/save/Assign_1Assignlayer/biases/Variable_1rnn_1/save/RestoreV2:1*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_1/save/Assign_2Assignlayer/weights/Variablernn_1/save/RestoreV2:2*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_1/save/Assign_3Assignlayer/weights/Variable_1rnn_1/save/RestoreV2:3*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_1/save/Assign_4Assignrnn/beta1_powerrnn_1/save/RestoreV2:4*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(
�
rnn_1/save/Assign_5Assignrnn/beta2_powerrnn_1/save/RestoreV2:5*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(
�
rnn_1/save/Assign_6Assignrnn/layer/biases/Variable/Adamrnn_1/save/RestoreV2:6*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_1/save/Assign_7Assign rnn/layer/biases/Variable/Adam_1rnn_1/save/RestoreV2:7*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_1/save/Assign_8Assign rnn/layer/biases/Variable_1/Adamrnn_1/save/RestoreV2:8*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_1/save/Assign_9Assign"rnn/layer/biases/Variable_1/Adam_1rnn_1/save/RestoreV2:9*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_1/save/Assign_10Assignrnn/layer/weights/Variable/Adamrnn_1/save/RestoreV2:10*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_1/save/Assign_11Assign!rnn/layer/weights/Variable/Adam_1rnn_1/save/RestoreV2:11*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_1/save/Assign_12Assign!rnn/layer/weights/Variable_1/Adamrnn_1/save/RestoreV2:12*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_1/save/Assign_13Assign#rnn/layer/weights/Variable_1/Adam_1rnn_1/save/RestoreV2:13*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_1/save/Assign_14Assignrnn/rnn/basic_lstm_cell/biasrnn_1/save/RestoreV2:14*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
rnn_1/save/Assign_15Assignrnn/rnn/basic_lstm_cell/kernelrnn_1/save/RestoreV2:15*
use_locking(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
rnn_1/save/Assign_16Assign%rnn/rnn/rnn/basic_lstm_cell/bias/Adamrnn_1/save/RestoreV2:16*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
rnn_1/save/Assign_17Assign'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1rnn_1/save/RestoreV2:17*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
rnn_1/save/Assign_18Assign'rnn/rnn/rnn/basic_lstm_cell/kernel/Adamrnn_1/save/RestoreV2:18*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
rnn_1/save/Assign_19Assign)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn_1/save/RestoreV2:19*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
rnn_1/save/restore_allNoOp^rnn_1/save/Assign^rnn_1/save/Assign_1^rnn_1/save/Assign_10^rnn_1/save/Assign_11^rnn_1/save/Assign_12^rnn_1/save/Assign_13^rnn_1/save/Assign_14^rnn_1/save/Assign_15^rnn_1/save/Assign_16^rnn_1/save/Assign_17^rnn_1/save/Assign_18^rnn_1/save/Assign_19^rnn_1/save/Assign_2^rnn_1/save/Assign_3^rnn_1/save/Assign_4^rnn_1/save/Assign_5^rnn_1/save/Assign_6^rnn_1/save/Assign_7^rnn_1/save/Assign_8^rnn_1/save/Assign_9
s
inputs_1Placeholder*
dtype0*+
_output_shapes
:���������* 
shape:���������
t
	outputs_1Placeholder*
dtype0*+
_output_shapes
:���������* 
shape:���������
t
#layer_1/weights/random_normal/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
g
"layer_1/weights/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
$layer_1/weights/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
2layer_1/weights/random_normal/RandomStandardNormalRandomStandardNormal#layer_1/weights/random_normal/shape*

seed *
T0*
dtype0*
_output_shapes

:
*
seed2 
�
!layer_1/weights/random_normal/mulMul2layer_1/weights/random_normal/RandomStandardNormal$layer_1/weights/random_normal/stddev*
T0*
_output_shapes

:

�
layer_1/weights/random_normalAdd!layer_1/weights/random_normal/mul"layer_1/weights/random_normal/mean*
T0*
_output_shapes

:

�
layer_1/weights/Variable
VariableV2*
shape
:
*
shared_name *
dtype0*
_output_shapes

:
*
	container 
�
layer_1/weights/Variable/AssignAssignlayer_1/weights/Variablelayer_1/weights/random_normal*
use_locking(*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:

�
layer_1/weights/Variable/readIdentitylayer_1/weights/Variable*
T0*+
_class!
loc:@layer_1/weights/Variable*
_output_shapes

:

v
%layer_1/weights/random_normal_1/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
i
$layer_1/weights/random_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
k
&layer_1/weights/random_normal_1/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
4layer_1/weights/random_normal_1/RandomStandardNormalRandomStandardNormal%layer_1/weights/random_normal_1/shape*

seed *
T0*
dtype0*
_output_shapes

:
*
seed2 
�
#layer_1/weights/random_normal_1/mulMul4layer_1/weights/random_normal_1/RandomStandardNormal&layer_1/weights/random_normal_1/stddev*
T0*
_output_shapes

:

�
layer_1/weights/random_normal_1Add#layer_1/weights/random_normal_1/mul$layer_1/weights/random_normal_1/mean*
T0*
_output_shapes

:

�
layer_1/weights/Variable_1
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
!layer_1/weights/Variable_1/AssignAssignlayer_1/weights/Variable_1layer_1/weights/random_normal_1*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
layer_1/weights/Variable_1/readIdentitylayer_1/weights/Variable_1*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
_output_shapes

:

a
layer_1/biases/ConstConst*
valueB
*���=*
dtype0*
_output_shapes
:

�
layer_1/biases/Variable
VariableV2*
shape:
*
shared_name *
dtype0*
_output_shapes
:
*
	container 
�
layer_1/biases/Variable/AssignAssignlayer_1/biases/Variablelayer_1/biases/Const*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
layer_1/biases/Variable/readIdentitylayer_1/biases/Variable*
T0**
_class 
loc:@layer_1/biases/Variable*
_output_shapes
:

c
layer_1/biases/Const_1Const*
valueB*���=*
dtype0*
_output_shapes
:
�
layer_1/biases/Variable_1
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
 layer_1/biases/Variable_1/AssignAssignlayer_1/biases/Variable_1layer_1/biases/Const_1*
use_locking(*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
layer_1/biases/Variable_1/readIdentitylayer_1/biases/Variable_1*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
_output_shapes
:
d
rnn_2/Reshape/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
w
rnn_2/ReshapeReshapeinputs_1rnn_2/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:���������
�
rnn_2/MatMulMatMulrnn_2/Reshapelayer_1/weights/Variable/read*
transpose_b( *
T0*'
_output_shapes
:���������
*
transpose_a( 
n
	rnn_2/addAddrnn_2/MatMullayer_1/biases/Variable/read*
T0*'
_output_shapes
:���������

j
rnn_2/Reshape_1/shapeConst*!
valueB"����   
   *
dtype0*
_output_shapes
:
�
rnn_2/Reshape_1Reshape	rnn_2/addrnn_2/Reshape_1/shape*
T0*
Tshape0*+
_output_shapes
:���������

l
"rnn_2/BasicLSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
n
$rnn_2/BasicLSTMCellZeroState/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
j
(rnn_2/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
#rnn_2/BasicLSTMCellZeroState/concatConcatV2"rnn_2/BasicLSTMCellZeroState/Const$rnn_2/BasicLSTMCellZeroState/Const_1(rnn_2/BasicLSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
m
(rnn_2/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"rnn_2/BasicLSTMCellZeroState/zerosFill#rnn_2/BasicLSTMCellZeroState/concat(rnn_2/BasicLSTMCellZeroState/zeros/Const*
T0*

index_type0*
_output_shapes

:

n
$rnn_2/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
n
$rnn_2/BasicLSTMCellZeroState/Const_3Const*
valueB:
*
dtype0*
_output_shapes
:
n
$rnn_2/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
n
$rnn_2/BasicLSTMCellZeroState/Const_5Const*
valueB:
*
dtype0*
_output_shapes
:
l
*rnn_2/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
%rnn_2/BasicLSTMCellZeroState/concat_1ConcatV2$rnn_2/BasicLSTMCellZeroState/Const_4$rnn_2/BasicLSTMCellZeroState/Const_5*rnn_2/BasicLSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
o
*rnn_2/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
$rnn_2/BasicLSTMCellZeroState/zeros_1Fill%rnn_2/BasicLSTMCellZeroState/concat_1*rnn_2/BasicLSTMCellZeroState/zeros_1/Const*
T0*

index_type0*
_output_shapes

:

n
$rnn_2/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
n
$rnn_2/BasicLSTMCellZeroState/Const_7Const*
valueB:
*
dtype0*
_output_shapes
:
P
rnn_2/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
W
rnn_2/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
W
rnn_2/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
~
rnn_2/rnn/rangeRangernn_2/rnn/range/startrnn_2/rnn/Rankrnn_2/rnn/range/delta*
_output_shapes
:*

Tidx0
j
rnn_2/rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
W
rnn_2/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_2/rnn/concatConcatV2rnn_2/rnn/concat/values_0rnn_2/rnn/rangernn_2/rnn/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
rnn_2/rnn/transpose	Transposernn_2/Reshape_1rnn_2/rnn/concat*
Tperm0*
T0*+
_output_shapes
:���������

b
rnn_2/rnn/ShapeShapernn_2/rnn/transpose*
T0*
out_type0*
_output_shapes
:
g
rnn_2/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
rnn_2/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
i
rnn_2/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
rnn_2/rnn/strided_sliceStridedSlicernn_2/rnn/Shapernn_2/rnn/strided_slice/stackrnn_2/rnn/strided_slice/stack_1rnn_2/rnn/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
d
rnn_2/rnn/Shape_1Shapernn_2/rnn/transpose*
T0*
out_type0*
_output_shapes
:
i
rnn_2/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
k
!rnn_2/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
k
!rnn_2/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
rnn_2/rnn/strided_slice_1StridedSlicernn_2/rnn/Shape_1rnn_2/rnn/strided_slice_1/stack!rnn_2/rnn/strided_slice_1/stack_1!rnn_2/rnn/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
d
rnn_2/rnn/Shape_2Shapernn_2/rnn/transpose*
T0*
out_type0*
_output_shapes
:
i
rnn_2/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
k
!rnn_2/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
k
!rnn_2/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
rnn_2/rnn/strided_slice_2StridedSlicernn_2/rnn/Shape_2rnn_2/rnn/strided_slice_2/stack!rnn_2/rnn/strided_slice_2/stack_1!rnn_2/rnn/strided_slice_2/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
Z
rnn_2/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_2/rnn/ExpandDims
ExpandDimsrnn_2/rnn/strided_slice_2rnn_2/rnn/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
Y
rnn_2/rnn/ConstConst*
valueB:
*
dtype0*
_output_shapes
:
Y
rnn_2/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_2/rnn/concat_1ConcatV2rnn_2/rnn/ExpandDimsrnn_2/rnn/Constrnn_2/rnn/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
Z
rnn_2/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
rnn_2/rnn/zerosFillrnn_2/rnn/concat_1rnn_2/rnn/zeros/Const*
T0*

index_type0*'
_output_shapes
:���������

P
rnn_2/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_2/rnn/TensorArrayTensorArrayV3rnn_2/rnn/strided_slice_1*5
tensor_array_name rnn_2/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *$
element_shape:���������
*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
�
rnn_2/rnn/TensorArray_1TensorArrayV3rnn_2/rnn/strided_slice_1*
identical_element_shapes(*4
tensor_array_namernn_2/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *$
element_shape:���������
*
dynamic_size( *
clear_after_read(
u
"rnn_2/rnn/TensorArrayUnstack/ShapeShapernn_2/rnn/transpose*
T0*
out_type0*
_output_shapes
:
z
0rnn_2/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2rnn_2/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2rnn_2/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
*rnn_2/rnn/TensorArrayUnstack/strided_sliceStridedSlice"rnn_2/rnn/TensorArrayUnstack/Shape0rnn_2/rnn/TensorArrayUnstack/strided_slice/stack2rnn_2/rnn/TensorArrayUnstack/strided_slice/stack_12rnn_2/rnn/TensorArrayUnstack/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
j
(rnn_2/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
j
(rnn_2/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
"rnn_2/rnn/TensorArrayUnstack/rangeRange(rnn_2/rnn/TensorArrayUnstack/range/start*rnn_2/rnn/TensorArrayUnstack/strided_slice(rnn_2/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:���������*

Tidx0
�
Drnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn_2/rnn/TensorArray_1"rnn_2/rnn/TensorArrayUnstack/rangernn_2/rnn/transposernn_2/rnn/TensorArray_1:1*
T0*&
_class
loc:@rnn_2/rnn/transpose*
_output_shapes
: 
U
rnn_2/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
m
rnn_2/rnn/MaximumMaximumrnn_2/rnn/Maximum/xrnn_2/rnn/strided_slice_1*
T0*
_output_shapes
: 
k
rnn_2/rnn/MinimumMinimumrnn_2/rnn/strided_slice_1rnn_2/rnn/Maximum*
T0*
_output_shapes
: 
c
!rnn_2/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_2/rnn/while/EnterEnter!rnn_2/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *-

frame_namernn_2/rnn/while/while_context
�
rnn_2/rnn/while/Enter_1Enterrnn_2/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *-

frame_namernn_2/rnn/while/while_context
�
rnn_2/rnn/while/Enter_2Enterrnn_2/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *-

frame_namernn_2/rnn/while/while_context
�
rnn_2/rnn/while/Enter_3Enter"rnn_2/BasicLSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*-

frame_namernn_2/rnn/while/while_context
�
rnn_2/rnn/while/Enter_4Enter$rnn_2/BasicLSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*-

frame_namernn_2/rnn/while/while_context
�
rnn_2/rnn/while/MergeMergernn_2/rnn/while/Enterrnn_2/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 
�
rnn_2/rnn/while/Merge_1Mergernn_2/rnn/while/Enter_1rnn_2/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
rnn_2/rnn/while/Merge_2Mergernn_2/rnn/while/Enter_2rnn_2/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
�
rnn_2/rnn/while/Merge_3Mergernn_2/rnn/while/Enter_3rnn_2/rnn/while/NextIteration_3*
T0*
N* 
_output_shapes
:
: 
�
rnn_2/rnn/while/Merge_4Mergernn_2/rnn/while/Enter_4rnn_2/rnn/while/NextIteration_4*
T0*
N* 
_output_shapes
:
: 
p
rnn_2/rnn/while/LessLessrnn_2/rnn/while/Mergernn_2/rnn/while/Less/Enter*
T0*
_output_shapes
: 
�
rnn_2/rnn/while/Less/EnterEnterrnn_2/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *-

frame_namernn_2/rnn/while/while_context
v
rnn_2/rnn/while/Less_1Lessrnn_2/rnn/while/Merge_1rnn_2/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
�
rnn_2/rnn/while/Less_1/EnterEnterrnn_2/rnn/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *-

frame_namernn_2/rnn/while/while_context
n
rnn_2/rnn/while/LogicalAnd
LogicalAndrnn_2/rnn/while/Lessrnn_2/rnn/while/Less_1*
_output_shapes
: 
X
rnn_2/rnn/while/LoopCondLoopCondrnn_2/rnn/while/LogicalAnd*
_output_shapes
: 
�
rnn_2/rnn/while/SwitchSwitchrnn_2/rnn/while/Mergernn_2/rnn/while/LoopCond*
T0*(
_class
loc:@rnn_2/rnn/while/Merge*
_output_shapes
: : 
�
rnn_2/rnn/while/Switch_1Switchrnn_2/rnn/while/Merge_1rnn_2/rnn/while/LoopCond*
T0**
_class 
loc:@rnn_2/rnn/while/Merge_1*
_output_shapes
: : 
�
rnn_2/rnn/while/Switch_2Switchrnn_2/rnn/while/Merge_2rnn_2/rnn/while/LoopCond*
T0**
_class 
loc:@rnn_2/rnn/while/Merge_2*
_output_shapes
: : 
�
rnn_2/rnn/while/Switch_3Switchrnn_2/rnn/while/Merge_3rnn_2/rnn/while/LoopCond*
T0**
_class 
loc:@rnn_2/rnn/while/Merge_3*(
_output_shapes
:
:

�
rnn_2/rnn/while/Switch_4Switchrnn_2/rnn/while/Merge_4rnn_2/rnn/while/LoopCond*
T0**
_class 
loc:@rnn_2/rnn/while/Merge_4*(
_output_shapes
:
:

_
rnn_2/rnn/while/IdentityIdentityrnn_2/rnn/while/Switch:1*
T0*
_output_shapes
: 
c
rnn_2/rnn/while/Identity_1Identityrnn_2/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
c
rnn_2/rnn/while/Identity_2Identityrnn_2/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
k
rnn_2/rnn/while/Identity_3Identityrnn_2/rnn/while/Switch_3:1*
T0*
_output_shapes

:

k
rnn_2/rnn/while/Identity_4Identityrnn_2/rnn/while/Switch_4:1*
T0*
_output_shapes

:

r
rnn_2/rnn/while/add/yConst^rnn_2/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
l
rnn_2/rnn/while/addAddrnn_2/rnn/while/Identityrnn_2/rnn/while/add/y*
T0*
_output_shapes
: 
�
!rnn_2/rnn/while/TensorArrayReadV3TensorArrayReadV3'rnn_2/rnn/while/TensorArrayReadV3/Enterrnn_2/rnn/while/Identity_1)rnn_2/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:���������

�
'rnn_2/rnn/while/TensorArrayReadV3/EnterEnterrnn_2/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_2/rnn/while/while_context
�
)rnn_2/rnn/while/TensorArrayReadV3/Enter_1EnterDrnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *-

frame_namernn_2/rnn/while/while_context
�
%rnn_2/rnn/while/basic_lstm_cell/ConstConst^rnn_2/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
+rnn_2/rnn/while/basic_lstm_cell/concat/axisConst^rnn_2/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
&rnn_2/rnn/while/basic_lstm_cell/concatConcatV2!rnn_2/rnn/while/TensorArrayReadV3rnn_2/rnn/while/Identity_4+rnn_2/rnn/while/basic_lstm_cell/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
�
&rnn_2/rnn/while/basic_lstm_cell/MatMulMatMul&rnn_2/rnn/while/basic_lstm_cell/concat,rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter*
transpose_b( *
T0*
_output_shapes

:(*
transpose_a( 
�
,rnn_2/rnn/while/basic_lstm_cell/MatMul/EnterEnter#rnn/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*-

frame_namernn_2/rnn/while/while_context
�
'rnn_2/rnn/while/basic_lstm_cell/BiasAddBiasAdd&rnn_2/rnn/while/basic_lstm_cell/MatMul-rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes

:(
�
-rnn_2/rnn/while/basic_lstm_cell/BiasAdd/EnterEnter!rnn/rnn/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:(*-

frame_namernn_2/rnn/while/while_context
�
'rnn_2/rnn/while/basic_lstm_cell/Const_1Const^rnn_2/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
%rnn_2/rnn/while/basic_lstm_cell/splitSplit%rnn_2/rnn/while/basic_lstm_cell/Const'rnn_2/rnn/while/basic_lstm_cell/BiasAdd*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split
�
'rnn_2/rnn/while/basic_lstm_cell/Const_2Const^rnn_2/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
#rnn_2/rnn/while/basic_lstm_cell/AddAdd'rnn_2/rnn/while/basic_lstm_cell/split:2'rnn_2/rnn/while/basic_lstm_cell/Const_2*
T0*
_output_shapes

:

�
'rnn_2/rnn/while/basic_lstm_cell/SigmoidSigmoid#rnn_2/rnn/while/basic_lstm_cell/Add*
T0*
_output_shapes

:

�
#rnn_2/rnn/while/basic_lstm_cell/MulMulrnn_2/rnn/while/Identity_3'rnn_2/rnn/while/basic_lstm_cell/Sigmoid*
T0*
_output_shapes

:

�
)rnn_2/rnn/while/basic_lstm_cell/Sigmoid_1Sigmoid%rnn_2/rnn/while/basic_lstm_cell/split*
T0*
_output_shapes

:

~
$rnn_2/rnn/while/basic_lstm_cell/TanhTanh'rnn_2/rnn/while/basic_lstm_cell/split:1*
T0*
_output_shapes

:

�
%rnn_2/rnn/while/basic_lstm_cell/Mul_1Mul)rnn_2/rnn/while/basic_lstm_cell/Sigmoid_1$rnn_2/rnn/while/basic_lstm_cell/Tanh*
T0*
_output_shapes

:

�
%rnn_2/rnn/while/basic_lstm_cell/Add_1Add#rnn_2/rnn/while/basic_lstm_cell/Mul%rnn_2/rnn/while/basic_lstm_cell/Mul_1*
T0*
_output_shapes

:

~
&rnn_2/rnn/while/basic_lstm_cell/Tanh_1Tanh%rnn_2/rnn/while/basic_lstm_cell/Add_1*
T0*
_output_shapes

:

�
)rnn_2/rnn/while/basic_lstm_cell/Sigmoid_2Sigmoid'rnn_2/rnn/while/basic_lstm_cell/split:3*
T0*
_output_shapes

:

�
%rnn_2/rnn/while/basic_lstm_cell/Mul_2Mul&rnn_2/rnn/while/basic_lstm_cell/Tanh_1)rnn_2/rnn/while/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes

:

�
3rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV39rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn_2/rnn/while/Identity_1%rnn_2/rnn/while/basic_lstm_cell/Mul_2rnn_2/rnn/while/Identity_2*
T0*8
_class.
,*loc:@rnn_2/rnn/while/basic_lstm_cell/Mul_2*
_output_shapes
: 
�
9rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn_2/rnn/TensorArray*
is_constant(*
_output_shapes
:*-

frame_namernn_2/rnn/while/while_context*
T0*8
_class.
,*loc:@rnn_2/rnn/while/basic_lstm_cell/Mul_2*
parallel_iterations 
t
rnn_2/rnn/while/add_1/yConst^rnn_2/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
r
rnn_2/rnn/while/add_1Addrnn_2/rnn/while/Identity_1rnn_2/rnn/while/add_1/y*
T0*
_output_shapes
: 
d
rnn_2/rnn/while/NextIterationNextIterationrnn_2/rnn/while/add*
T0*
_output_shapes
: 
h
rnn_2/rnn/while/NextIteration_1NextIterationrnn_2/rnn/while/add_1*
T0*
_output_shapes
: 
�
rnn_2/rnn/while/NextIteration_2NextIteration3rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
rnn_2/rnn/while/NextIteration_3NextIteration%rnn_2/rnn/while/basic_lstm_cell/Add_1*
T0*
_output_shapes

:

�
rnn_2/rnn/while/NextIteration_4NextIteration%rnn_2/rnn/while/basic_lstm_cell/Mul_2*
T0*
_output_shapes

:

U
rnn_2/rnn/while/ExitExitrnn_2/rnn/while/Switch*
T0*
_output_shapes
: 
Y
rnn_2/rnn/while/Exit_1Exitrnn_2/rnn/while/Switch_1*
T0*
_output_shapes
: 
Y
rnn_2/rnn/while/Exit_2Exitrnn_2/rnn/while/Switch_2*
T0*
_output_shapes
: 
a
rnn_2/rnn/while/Exit_3Exitrnn_2/rnn/while/Switch_3*
T0*
_output_shapes

:

a
rnn_2/rnn/while/Exit_4Exitrnn_2/rnn/while/Switch_4*
T0*
_output_shapes

:

�
,rnn_2/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn_2/rnn/TensorArrayrnn_2/rnn/while/Exit_2*(
_class
loc:@rnn_2/rnn/TensorArray*
_output_shapes
: 
�
&rnn_2/rnn/TensorArrayStack/range/startConst*(
_class
loc:@rnn_2/rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 
�
&rnn_2/rnn/TensorArrayStack/range/deltaConst*(
_class
loc:@rnn_2/rnn/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
�
 rnn_2/rnn/TensorArrayStack/rangeRange&rnn_2/rnn/TensorArrayStack/range/start,rnn_2/rnn/TensorArrayStack/TensorArraySizeV3&rnn_2/rnn/TensorArrayStack/range/delta*

Tidx0*(
_class
loc:@rnn_2/rnn/TensorArray*#
_output_shapes
:���������
�
.rnn_2/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn_2/rnn/TensorArray rnn_2/rnn/TensorArrayStack/rangernn_2/rnn/while/Exit_2*(
_class
loc:@rnn_2/rnn/TensorArray*
dtype0*"
_output_shapes
:
*
element_shape
:

[
rnn_2/rnn/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
R
rnn_2/rnn/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
Y
rnn_2/rnn/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
Y
rnn_2/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
rnn_2/rnn/range_1Rangernn_2/rnn/range_1/startrnn_2/rnn/Rank_1rnn_2/rnn/range_1/delta*
_output_shapes
:*

Tidx0
l
rnn_2/rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
Y
rnn_2/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_2/rnn/concat_2ConcatV2rnn_2/rnn/concat_2/values_0rnn_2/rnn/range_1rnn_2/rnn/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
rnn_2/rnn/transpose_1	Transpose.rnn_2/rnn/TensorArrayStack/TensorArrayGatherV3rnn_2/rnn/concat_2*
Tperm0*
T0*"
_output_shapes
:

f
rnn_2/Reshape_2/shapeConst*
valueB"����
   *
dtype0*
_output_shapes
:

rnn_2/Reshape_2Reshapernn_2/rnn/transpose_1rnn_2/Reshape_2/shape*
T0*
Tshape0*
_output_shapes

:


�
rnn_2/MatMul_1MatMulrnn_2/Reshape_2layer_1/weights/Variable_1/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b( 
k
rnn_2/predsAddrnn_2/MatMul_1layer_1/biases/Variable_1/read*
T0*
_output_shapes

:

h
rnn_2/Reshape_3/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
q
rnn_2/Reshape_3Reshapernn_2/predsrnn_2/Reshape_3/shape*
T0*
Tshape0*
_output_shapes
:

h
rnn_2/Reshape_4/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
x
rnn_2/Reshape_4Reshape	outputs_1rnn_2/Reshape_4/shape*
T0*
Tshape0*#
_output_shapes
:���������
W
	rnn_2/subSubrnn_2/Reshape_3rnn_2/Reshape_4*
T0*
_output_shapes
:

F
rnn_2/SquareSquare	rnn_2/sub*
T0*
_output_shapes
:

U
rnn_2/ConstConst*
valueB: *
dtype0*
_output_shapes
:
k

rnn_2/MeanMeanrnn_2/Squarernn_2/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
X
rnn_2/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
rnn_2/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
rnn_2/gradients/FillFillrnn_2/gradients/Shapernn_2/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
Y
rnn_2/gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_2/gradients/f_count_1Enterrnn_2/gradients/f_count*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *-

frame_namernn_2/rnn/while/while_context
�
rnn_2/gradients/MergeMergernn_2/gradients/f_count_1rnn_2/gradients/NextIteration*
T0*
N*
_output_shapes
: : 
t
rnn_2/gradients/SwitchSwitchrnn_2/gradients/Mergernn_2/rnn/while/LoopCond*
T0*
_output_shapes
: : 
r
rnn_2/gradients/Add/yConst^rnn_2/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
l
rnn_2/gradients/AddAddrnn_2/gradients/Switch:1rnn_2/gradients/Add/y*
T0*
_output_shapes
: 
�
rnn_2/gradients/NextIterationNextIterationrnn_2/gradients/Addg^rnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2Q^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2K^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2M^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2K^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2M^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2I^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2K^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2O^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2Q^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1*
T0*
_output_shapes
: 
Z
rnn_2/gradients/f_count_2Exitrnn_2/gradients/Switch*
T0*
_output_shapes
: 
Y
rnn_2/gradients/b_countConst*
value	B :*
dtype0*
_output_shapes
: 
�
rnn_2/gradients/b_count_1Enterrnn_2/gradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
rnn_2/gradients/Merge_1Mergernn_2/gradients/b_count_1rnn_2/gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
rnn_2/gradients/GreaterEqualGreaterEqualrnn_2/gradients/Merge_1"rnn_2/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
�
"rnn_2/gradients/GreaterEqual/EnterEnterrnn_2/gradients/b_count*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
[
rnn_2/gradients/b_count_2LoopCondrnn_2/gradients/GreaterEqual*
_output_shapes
: 
y
rnn_2/gradients/Switch_1Switchrnn_2/gradients/Merge_1rnn_2/gradients/b_count_2*
T0*
_output_shapes
: : 
{
rnn_2/gradients/SubSubrnn_2/gradients/Switch_1:1"rnn_2/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
�
rnn_2/gradients/NextIteration_1NextIterationrnn_2/gradients/Subb^rnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
\
rnn_2/gradients/b_count_3Exitrnn_2/gradients/Switch_1*
T0*
_output_shapes
: 
w
-rnn_2/gradients/rnn_2/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
'rnn_2/gradients/rnn_2/Mean_grad/ReshapeReshapernn_2/gradients/Fill-rnn_2/gradients/rnn_2/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
o
%rnn_2/gradients/rnn_2/Mean_grad/ConstConst*
valueB:
*
dtype0*
_output_shapes
:
�
$rnn_2/gradients/rnn_2/Mean_grad/TileTile'rnn_2/gradients/rnn_2/Mean_grad/Reshape%rnn_2/gradients/rnn_2/Mean_grad/Const*

Tmultiples0*
T0*
_output_shapes
:

l
'rnn_2/gradients/rnn_2/Mean_grad/Const_1Const*
valueB
 *   A*
dtype0*
_output_shapes
: 
�
'rnn_2/gradients/rnn_2/Mean_grad/truedivRealDiv$rnn_2/gradients/rnn_2/Mean_grad/Tile'rnn_2/gradients/rnn_2/Mean_grad/Const_1*
T0*
_output_shapes
:

�
'rnn_2/gradients/rnn_2/Square_grad/ConstConst(^rnn_2/gradients/rnn_2/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
%rnn_2/gradients/rnn_2/Square_grad/MulMul	rnn_2/sub'rnn_2/gradients/rnn_2/Square_grad/Const*
T0*
_output_shapes
:

�
'rnn_2/gradients/rnn_2/Square_grad/Mul_1Mul'rnn_2/gradients/rnn_2/Mean_grad/truediv%rnn_2/gradients/rnn_2/Square_grad/Mul*
T0*
_output_shapes
:

n
$rnn_2/gradients/rnn_2/sub_grad/ShapeConst*
valueB:
*
dtype0*
_output_shapes
:
u
&rnn_2/gradients/rnn_2/sub_grad/Shape_1Shapernn_2/Reshape_4*
T0*
out_type0*
_output_shapes
:
�
4rnn_2/gradients/rnn_2/sub_grad/BroadcastGradientArgsBroadcastGradientArgs$rnn_2/gradients/rnn_2/sub_grad/Shape&rnn_2/gradients/rnn_2/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
"rnn_2/gradients/rnn_2/sub_grad/SumSum'rnn_2/gradients/rnn_2/Square_grad/Mul_14rnn_2/gradients/rnn_2/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
&rnn_2/gradients/rnn_2/sub_grad/ReshapeReshape"rnn_2/gradients/rnn_2/sub_grad/Sum$rnn_2/gradients/rnn_2/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:

�
$rnn_2/gradients/rnn_2/sub_grad/Sum_1Sum'rnn_2/gradients/rnn_2/Square_grad/Mul_16rnn_2/gradients/rnn_2/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
r
"rnn_2/gradients/rnn_2/sub_grad/NegNeg$rnn_2/gradients/rnn_2/sub_grad/Sum_1*
T0*
_output_shapes
:
�
(rnn_2/gradients/rnn_2/sub_grad/Reshape_1Reshape"rnn_2/gradients/rnn_2/sub_grad/Neg&rnn_2/gradients/rnn_2/sub_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������
�
/rnn_2/gradients/rnn_2/sub_grad/tuple/group_depsNoOp'^rnn_2/gradients/rnn_2/sub_grad/Reshape)^rnn_2/gradients/rnn_2/sub_grad/Reshape_1
�
7rnn_2/gradients/rnn_2/sub_grad/tuple/control_dependencyIdentity&rnn_2/gradients/rnn_2/sub_grad/Reshape0^rnn_2/gradients/rnn_2/sub_grad/tuple/group_deps*
T0*9
_class/
-+loc:@rnn_2/gradients/rnn_2/sub_grad/Reshape*
_output_shapes
:

�
9rnn_2/gradients/rnn_2/sub_grad/tuple/control_dependency_1Identity(rnn_2/gradients/rnn_2/sub_grad/Reshape_10^rnn_2/gradients/rnn_2/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@rnn_2/gradients/rnn_2/sub_grad/Reshape_1*#
_output_shapes
:���������
{
*rnn_2/gradients/rnn_2/Reshape_3_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
�
,rnn_2/gradients/rnn_2/Reshape_3_grad/ReshapeReshape7rnn_2/gradients/rnn_2/sub_grad/tuple/control_dependency*rnn_2/gradients/rnn_2/Reshape_3_grad/Shape*
T0*
Tshape0*
_output_shapes

:

w
&rnn_2/gradients/rnn_2/preds_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
r
(rnn_2/gradients/rnn_2/preds_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6rnn_2/gradients/rnn_2/preds_grad/BroadcastGradientArgsBroadcastGradientArgs&rnn_2/gradients/rnn_2/preds_grad/Shape(rnn_2/gradients/rnn_2/preds_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$rnn_2/gradients/rnn_2/preds_grad/SumSum,rnn_2/gradients/rnn_2/Reshape_3_grad/Reshape6rnn_2/gradients/rnn_2/preds_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

�
(rnn_2/gradients/rnn_2/preds_grad/ReshapeReshape$rnn_2/gradients/rnn_2/preds_grad/Sum&rnn_2/gradients/rnn_2/preds_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
&rnn_2/gradients/rnn_2/preds_grad/Sum_1Sum,rnn_2/gradients/rnn_2/Reshape_3_grad/Reshape8rnn_2/gradients/rnn_2/preds_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
*rnn_2/gradients/rnn_2/preds_grad/Reshape_1Reshape&rnn_2/gradients/rnn_2/preds_grad/Sum_1(rnn_2/gradients/rnn_2/preds_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1rnn_2/gradients/rnn_2/preds_grad/tuple/group_depsNoOp)^rnn_2/gradients/rnn_2/preds_grad/Reshape+^rnn_2/gradients/rnn_2/preds_grad/Reshape_1
�
9rnn_2/gradients/rnn_2/preds_grad/tuple/control_dependencyIdentity(rnn_2/gradients/rnn_2/preds_grad/Reshape2^rnn_2/gradients/rnn_2/preds_grad/tuple/group_deps*
T0*;
_class1
/-loc:@rnn_2/gradients/rnn_2/preds_grad/Reshape*
_output_shapes

:

�
;rnn_2/gradients/rnn_2/preds_grad/tuple/control_dependency_1Identity*rnn_2/gradients/rnn_2/preds_grad/Reshape_12^rnn_2/gradients/rnn_2/preds_grad/tuple/group_deps*
T0*=
_class3
1/loc:@rnn_2/gradients/rnn_2/preds_grad/Reshape_1*
_output_shapes
:
�
*rnn_2/gradients/rnn_2/MatMul_1_grad/MatMulMatMul9rnn_2/gradients/rnn_2/preds_grad/tuple/control_dependencylayer_1/weights/Variable_1/read*
transpose_b(*
T0*
_output_shapes

:

*
transpose_a( 
�
,rnn_2/gradients/rnn_2/MatMul_1_grad/MatMul_1MatMulrnn_2/Reshape_29rnn_2/gradients/rnn_2/preds_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_a(*
transpose_b( 
�
4rnn_2/gradients/rnn_2/MatMul_1_grad/tuple/group_depsNoOp+^rnn_2/gradients/rnn_2/MatMul_1_grad/MatMul-^rnn_2/gradients/rnn_2/MatMul_1_grad/MatMul_1
�
<rnn_2/gradients/rnn_2/MatMul_1_grad/tuple/control_dependencyIdentity*rnn_2/gradients/rnn_2/MatMul_1_grad/MatMul5^rnn_2/gradients/rnn_2/MatMul_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@rnn_2/gradients/rnn_2/MatMul_1_grad/MatMul*
_output_shapes

:


�
>rnn_2/gradients/rnn_2/MatMul_1_grad/tuple/control_dependency_1Identity,rnn_2/gradients/rnn_2/MatMul_1_grad/MatMul_15^rnn_2/gradients/rnn_2/MatMul_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@rnn_2/gradients/rnn_2/MatMul_1_grad/MatMul_1*
_output_shapes

:


*rnn_2/gradients/rnn_2/Reshape_2_grad/ShapeConst*!
valueB"      
   *
dtype0*
_output_shapes
:
�
,rnn_2/gradients/rnn_2/Reshape_2_grad/ReshapeReshape<rnn_2/gradients/rnn_2/MatMul_1_grad/tuple/control_dependency*rnn_2/gradients/rnn_2/Reshape_2_grad/Shape*
T0*
Tshape0*"
_output_shapes
:

�
<rnn_2/gradients/rnn_2/rnn/transpose_1_grad/InvertPermutationInvertPermutationrnn_2/rnn/concat_2*
T0*
_output_shapes
:
�
4rnn_2/gradients/rnn_2/rnn/transpose_1_grad/transpose	Transpose,rnn_2/gradients/rnn_2/Reshape_2_grad/Reshape<rnn_2/gradients/rnn_2/rnn/transpose_1_grad/InvertPermutation*
Tperm0*
T0*"
_output_shapes
:

�
ernn_2/gradients/rnn_2/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn_2/rnn/TensorArrayrnn_2/rnn/while/Exit_2*(
_class
loc:@rnn_2/rnn/TensorArray*
sourcernn_2/gradients*
_output_shapes

:: 
�
arnn_2/gradients/rnn_2/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityrnn_2/rnn/while/Exit_2f^rnn_2/gradients/rnn_2/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*(
_class
loc:@rnn_2/rnn/TensorArray*
_output_shapes
: 
�
krnn_2/gradients/rnn_2/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3ernn_2/gradients/rnn_2/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3 rnn_2/rnn/TensorArrayStack/range4rnn_2/gradients/rnn_2/rnn/transpose_1_grad/transposearnn_2/gradients/rnn_2/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
j
rnn_2/gradients/zerosConst*
valueB
*    *
dtype0*
_output_shapes

:

l
rnn_2/gradients/zeros_1Const*
valueB
*    *
dtype0*
_output_shapes

:

�
2rnn_2/gradients/rnn_2/rnn/while/Exit_2_grad/b_exitEnterkrnn_2/gradients/rnn_2/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
2rnn_2/gradients/rnn_2/rnn/while/Exit_3_grad/b_exitEnterrnn_2/gradients/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
2rnn_2/gradients/rnn_2/rnn/while/Exit_4_grad/b_exitEnterrnn_2/gradients/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
6rnn_2/gradients/rnn_2/rnn/while/Switch_2_grad/b_switchMerge2rnn_2/gradients/rnn_2/rnn/while/Exit_2_grad/b_exit=rnn_2/gradients/rnn_2/rnn/while/Switch_2_grad_1/NextIteration*
T0*
N*
_output_shapes
: : 
�
6rnn_2/gradients/rnn_2/rnn/while/Switch_3_grad/b_switchMerge2rnn_2/gradients/rnn_2/rnn/while/Exit_3_grad/b_exit=rnn_2/gradients/rnn_2/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N* 
_output_shapes
:
: 
�
6rnn_2/gradients/rnn_2/rnn/while/Switch_4_grad/b_switchMerge2rnn_2/gradients/rnn_2/rnn/while/Exit_4_grad/b_exit=rnn_2/gradients/rnn_2/rnn/while/Switch_4_grad_1/NextIteration*
T0*
N* 
_output_shapes
:
: 
�
3rnn_2/gradients/rnn_2/rnn/while/Merge_2_grad/SwitchSwitch6rnn_2/gradients/rnn_2/rnn/while/Switch_2_grad/b_switchrnn_2/gradients/b_count_2*
T0*I
_class?
=;loc:@rnn_2/gradients/rnn_2/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: : 
{
=rnn_2/gradients/rnn_2/rnn/while/Merge_2_grad/tuple/group_depsNoOp4^rnn_2/gradients/rnn_2/rnn/while/Merge_2_grad/Switch
�
Ernn_2/gradients/rnn_2/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity3rnn_2/gradients/rnn_2/rnn/while/Merge_2_grad/Switch>^rnn_2/gradients/rnn_2/rnn/while/Merge_2_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_2/gradients/rnn_2/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
�
Grnn_2/gradients/rnn_2/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity5rnn_2/gradients/rnn_2/rnn/while/Merge_2_grad/Switch:1>^rnn_2/gradients/rnn_2/rnn/while/Merge_2_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_2/gradients/rnn_2/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
�
3rnn_2/gradients/rnn_2/rnn/while/Merge_3_grad/SwitchSwitch6rnn_2/gradients/rnn_2/rnn/while/Switch_3_grad/b_switchrnn_2/gradients/b_count_2*
T0*I
_class?
=;loc:@rnn_2/gradients/rnn_2/rnn/while/Switch_3_grad/b_switch*(
_output_shapes
:
:

{
=rnn_2/gradients/rnn_2/rnn/while/Merge_3_grad/tuple/group_depsNoOp4^rnn_2/gradients/rnn_2/rnn/while/Merge_3_grad/Switch
�
Ernn_2/gradients/rnn_2/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity3rnn_2/gradients/rnn_2/rnn/while/Merge_3_grad/Switch>^rnn_2/gradients/rnn_2/rnn/while/Merge_3_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_2/gradients/rnn_2/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
Grnn_2/gradients/rnn_2/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity5rnn_2/gradients/rnn_2/rnn/while/Merge_3_grad/Switch:1>^rnn_2/gradients/rnn_2/rnn/while/Merge_3_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_2/gradients/rnn_2/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
3rnn_2/gradients/rnn_2/rnn/while/Merge_4_grad/SwitchSwitch6rnn_2/gradients/rnn_2/rnn/while/Switch_4_grad/b_switchrnn_2/gradients/b_count_2*
T0*I
_class?
=;loc:@rnn_2/gradients/rnn_2/rnn/while/Switch_4_grad/b_switch*(
_output_shapes
:
:

{
=rnn_2/gradients/rnn_2/rnn/while/Merge_4_grad/tuple/group_depsNoOp4^rnn_2/gradients/rnn_2/rnn/while/Merge_4_grad/Switch
�
Ernn_2/gradients/rnn_2/rnn/while/Merge_4_grad/tuple/control_dependencyIdentity3rnn_2/gradients/rnn_2/rnn/while/Merge_4_grad/Switch>^rnn_2/gradients/rnn_2/rnn/while/Merge_4_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_2/gradients/rnn_2/rnn/while/Switch_4_grad/b_switch*
_output_shapes

:

�
Grnn_2/gradients/rnn_2/rnn/while/Merge_4_grad/tuple/control_dependency_1Identity5rnn_2/gradients/rnn_2/rnn/while/Merge_4_grad/Switch:1>^rnn_2/gradients/rnn_2/rnn/while/Merge_4_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_2/gradients/rnn_2/rnn/while/Switch_4_grad/b_switch*
_output_shapes

:

�
1rnn_2/gradients/rnn_2/rnn/while/Enter_2_grad/ExitExitErnn_2/gradients/rnn_2/rnn/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes
: 
�
1rnn_2/gradients/rnn_2/rnn/while/Enter_3_grad/ExitExitErnn_2/gradients/rnn_2/rnn/while/Merge_3_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
1rnn_2/gradients/rnn_2/rnn/while/Enter_4_grad/ExitExitErnn_2/gradients/rnn_2/rnn/while/Merge_4_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
jrnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3prnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterGrnn_2/gradients/rnn_2/rnn/while/Merge_2_grad/tuple/control_dependency_1*8
_class.
,*loc:@rnn_2/rnn/while/basic_lstm_cell/Mul_2*
sourcernn_2/gradients*
_output_shapes

:: 
�
prnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn_2/rnn/TensorArray*
is_constant(*
_output_shapes
:*=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context*
T0*8
_class.
,*loc:@rnn_2/rnn/while/basic_lstm_cell/Mul_2*
parallel_iterations 
�
frnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityGrnn_2/gradients/rnn_2/rnn/while/Merge_2_grad/tuple/control_dependency_1k^rnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*8
_class.
,*loc:@rnn_2/rnn/while/basic_lstm_cell/Mul_2*
_output_shapes
: 
�
Zrnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3jrnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3ernn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2frnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*'
_output_shapes
:���������

�
`rnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*-
_class#
!loc:@rnn_2/rnn/while/Identity_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
`rnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2`rnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*
	elem_type0*-
_class#
!loc:@rnn_2/rnn/while/Identity_1*

stack_name *
_output_shapes
:
�
`rnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnter`rnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_2/rnn/while/while_context
�
frnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2`rnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterrnn_2/rnn/while/Identity_1^rnn_2/gradients/Add*
T0*
_output_shapes
: *
swap_memory( 
�
ernn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2krnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^rnn_2/gradients/Sub*
_output_shapes
: *
	elem_type0
�
krnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnter`rnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
arnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerf^rnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2P^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2J^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2L^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2J^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2L^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2H^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2J^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2N^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2P^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
�
Yrnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpH^rnn_2/gradients/rnn_2/rnn/while/Merge_2_grad/tuple/control_dependency_1[^rnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
arnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityZrnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3Z^rnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*m
_classc
a_loc:@rnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
_output_shapes

:

�
crnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityGrnn_2/gradients/rnn_2/rnn/while/Merge_2_grad/tuple/control_dependency_1Z^rnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_2/gradients/rnn_2/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
�
rnn_2/gradients/AddNAddNGrnn_2/gradients/rnn_2/rnn/while/Merge_4_grad/tuple/control_dependency_1arnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*I
_class?
=;loc:@rnn_2/gradients/rnn_2/rnn/while/Switch_4_grad/b_switch*
N*
_output_shapes

:

�
>rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/MulMulrnn_2/gradients/AddNIrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2*
T0*
_output_shapes

:

�
Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/ConstConst*<
_class2
0.loc:@rnn_2/rnn/while/basic_lstm_cell/Sigmoid_2*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_accStackV2Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Const*
	elem_type0*<
_class2
0.loc:@rnn_2/rnn/while/basic_lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:
�
Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/EnterEnterDrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_2/rnn/while/while_context
�
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2StackPushV2Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Enter)rnn_2/rnn/while/basic_lstm_cell/Sigmoid_2^rnn_2/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Irnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2
StackPopV2Ornn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/Enter^rnn_2/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Ornn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/EnterEnterDrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1Mulrnn_2/gradients/AddNKrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2*
T0*
_output_shapes

:

�
Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/ConstConst*9
_class/
-+loc:@rnn_2/rnn/while/basic_lstm_cell/Tanh_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_accStackV2Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Const*
	elem_type0*9
_class/
-+loc:@rnn_2/rnn/while/basic_lstm_cell/Tanh_1*

stack_name *
_output_shapes
:
�
Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/EnterEnterFrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_2/rnn/while/while_context
�
Lrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2StackPushV2Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter&rnn_2/rnn/while/basic_lstm_cell/Tanh_1^rnn_2/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Krnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2
StackPopV2Qrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/Enter^rnn_2/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Qrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/EnterEnterFrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
Krnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_depsNoOp?^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/MulA^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1
�
Srnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependencyIdentity>rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/MulL^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul*
_output_shapes

:

�
Urnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1Identity@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1L^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*S
_classI
GEloc:@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1*
_output_shapes

:

�
Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradKrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2Srnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradIrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2Urnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
=rnn_2/gradients/rnn_2/rnn/while/Switch_2_grad_1/NextIterationNextIterationcrnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
rnn_2/gradients/AddN_1AddNGrnn_2/gradients/rnn_2/rnn/while/Merge_3_grad/tuple/control_dependency_1Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*I
_class?
=;loc:@rnn_2/gradients/rnn_2/rnn/while/Switch_3_grad/b_switch*
N*
_output_shapes

:

l
Krnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_depsNoOp^rnn_2/gradients/AddN_1
�
Srnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyIdentityrnn_2/gradients/AddN_1L^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_2/gradients/rnn_2/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
Urnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Identityrnn_2/gradients/AddN_1L^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_2/gradients/rnn_2/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
<rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/MulMulSrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyGrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2*
T0*
_output_shapes

:

�
Brnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/ConstConst*:
_class0
.,loc:@rnn_2/rnn/while/basic_lstm_cell/Sigmoid*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Brnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_accStackV2Brnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/Const*:
_class0
.,loc:@rnn_2/rnn/while/basic_lstm_cell/Sigmoid*

stack_name *
_output_shapes
:*
	elem_type0
�
Brnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/EnterEnterBrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_2/rnn/while/while_context
�
Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2StackPushV2Brnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/Enter'rnn_2/rnn/while/basic_lstm_cell/Sigmoid^rnn_2/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Grnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2
StackPopV2Mrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/Enter^rnn_2/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Mrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/EnterEnterBrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
>rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1MulSrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyIrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2*
T0*
_output_shapes

:

�
Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/ConstConst*-
_class#
!loc:@rnn_2/rnn/while/Identity_3*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_accStackV2Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Const*-
_class#
!loc:@rnn_2/rnn/while/Identity_3*

stack_name *
_output_shapes
:*
	elem_type0
�
Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/EnterEnterDrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_2/rnn/while/while_context
�
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2StackPushV2Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Enterrnn_2/rnn/while/Identity_3^rnn_2/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Irnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
StackPopV2Ornn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/Enter^rnn_2/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Ornn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/EnterEnterDrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
Irnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_depsNoOp=^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul?^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1
�
Qrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependencyIdentity<rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/MulJ^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul*
_output_shapes

:

�
Srnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1Identity>rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1J^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1*
_output_shapes

:

�
>rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/MulMulUrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Irnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2*
T0*
_output_shapes

:

�
Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/ConstConst*7
_class-
+)loc:@rnn_2/rnn/while/basic_lstm_cell/Tanh*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_accStackV2Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Const*7
_class-
+)loc:@rnn_2/rnn/while/basic_lstm_cell/Tanh*

stack_name *
_output_shapes
:*
	elem_type0
�
Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/EnterEnterDrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_2/rnn/while/while_context
�
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2StackPushV2Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Enter$rnn_2/rnn/while/basic_lstm_cell/Tanh^rnn_2/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Irnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2
StackPopV2Ornn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/Enter^rnn_2/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Ornn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/EnterEnterDrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1MulUrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Krnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2*
T0*
_output_shapes

:

�
Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/ConstConst*<
_class2
0.loc:@rnn_2/rnn/while/basic_lstm_cell/Sigmoid_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_accStackV2Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Const*
	elem_type0*<
_class2
0.loc:@rnn_2/rnn/while/basic_lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:
�
Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/EnterEnterFrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_2/rnn/while/while_context
�
Lrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2StackPushV2Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter)rnn_2/rnn/while/basic_lstm_cell/Sigmoid_1^rnn_2/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Krnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2
StackPopV2Qrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/Enter^rnn_2/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Qrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/EnterEnterFrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
Krnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_depsNoOp?^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/MulA^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1
�
Srnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependencyIdentity>rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/MulL^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul*
_output_shapes

:

�
Urnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1Identity@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1L^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*S
_classI
GEloc:@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1*
_output_shapes

:

�
Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradGrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2Srnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradKrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2Srnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
Brnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Tanh_grad/TanhGradTanhGradIrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2Urnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
=rnn_2/gradients/rnn_2/rnn/while/Switch_3_grad_1/NextIterationNextIterationQrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
>rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/ShapeConst^rnn_2/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/Shape_1Const^rnn_2/gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
�
Nrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgs>rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/Shape@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/SumSumHrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradNrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes

:
*
	keep_dims( *

Tidx0
�
@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/ReshapeReshape<rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/Sum>rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
>rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/Sum_1SumHrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradPrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Brnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/Reshape_1Reshape>rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/Sum_1@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Irnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/tuple/group_depsNoOpA^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/ReshapeC^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/Reshape_1
�
Qrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependencyIdentity@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/ReshapeJ^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*S
_classI
GEloc:@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/Reshape*
_output_shapes

:

�
Srnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependency_1IdentityBrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/Reshape_1J^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*U
_classK
IGloc:@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/Reshape_1*
_output_shapes
: 
�
Arnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/split_grad/concatConcatV2Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradBrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Tanh_grad/TanhGradQrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependencyJrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradGrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/split_grad/concat/Const*
T0*
N*
_output_shapes

:(*

Tidx0
�
Grnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/split_grad/concat/ConstConst^rnn_2/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradArnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:(
�
Mrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOpI^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradB^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/split_grad/concat
�
Urnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityArnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/split_grad/concatN^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/split_grad/concat*
_output_shapes

:(
�
Wrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityHrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradN^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes
:(
�
Brnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMulMatMulUrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyHrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul/Enter*
T0*
_output_shapes

:*
transpose_a( *
transpose_b(
�
Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter#rnn/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1MatMulOrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Urnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:(*
transpose_a(
�
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/ConstConst*9
_class/
-+loc:@rnn_2/rnn/while/basic_lstm_cell/concat*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Const*
	elem_type0*9
_class/
-+loc:@rnn_2/rnn/while/basic_lstm_cell/concat*

stack_name *
_output_shapes
:
�
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnterJrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_2/rnn/while/while_context
�
Prnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter&rnn_2/rnn/while/basic_lstm_cell/concat^rnn_2/gradients/Add*
T0*
_output_shapes

:*
swap_memory( 
�
Ornn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Urnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^rnn_2/gradients/Sub*
_output_shapes

:*
	elem_type0
�
Urnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterJrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
Lrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOpC^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMulE^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1
�
Trnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentityBrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMulM^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul*
_output_shapes

:
�
Vrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityDrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1M^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1*
_output_shapes

:(
�
Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB(*    *
dtype0*
_output_shapes
:(
�
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterHrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:(*=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeJrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Prnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes

:(: 
�
Irnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchJrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2rnn_2/gradients/b_count_2*
T0* 
_output_shapes
:(:(
�
Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/AddAddKrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1Wrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:(
�
Prnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationFrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes
:(
�
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitIrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes
:(
�
Arnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ConstConst^rnn_2/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/RankConst^rnn_2/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
?rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/modFloorModArnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/Const@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
�
Arnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeShape!rnn_2/rnn/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
�
Brnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeNShapeNMrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2Ornn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1*
T0*
out_type0*
N* 
_output_shapes
::
�
Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/ConstConst*4
_class*
(&loc:@rnn_2/rnn/while/TensorArrayReadV3*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_accStackV2Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const*4
_class*
(&loc:@rnn_2/rnn/while/TensorArrayReadV3*

stack_name *
_output_shapes
:*
	elem_type0
�
Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/EnterEnterHrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_2/rnn/while/while_context
�
Nrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter!rnn_2/rnn/while/TensorArrayReadV3^rnn_2/gradients/Add*
T0*'
_output_shapes
:���������
*
swap_memory( 
�
Mrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2Srnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^rnn_2/gradients/Sub*
	elem_type0*'
_output_shapes
:���������

�
Srnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterHrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const_1Const*-
_class#
!loc:@rnn_2/rnn/while/Identity_4*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1StackV2Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const_1*-
_class#
!loc:@rnn_2/rnn/while/Identity_4*

stack_name *
_output_shapes
:*
	elem_type0
�
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1EnterJrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_2/rnn/while/while_context
�
Prnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1StackPushV2Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1rnn_2/rnn/while/Identity_4^rnn_2/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Ornn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
StackPopV2Urnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/Enter^rnn_2/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Urnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/EnterEnterJrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffset?rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/modBrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeNDrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
�
Arnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/SliceSliceTrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyHrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ConcatOffsetBrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN*
Index0*
T0*'
_output_shapes
:���������

�
Crnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/Slice_1SliceTrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyJrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ConcatOffset:1Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*
_output_shapes

:

�
Lrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/tuple/group_depsNoOpB^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/SliceD^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/Slice_1
�
Trnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependencyIdentityArnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/SliceM^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/Slice*'
_output_shapes
:���������

�
Vrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1IdentityCrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/Slice_1M^rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*V
_classL
JHloc:@rnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes

:

�
Grnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB(*    *
dtype0*
_output_shapes

:(
�
Irnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1EnterGrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:(*=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
Irnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2MergeIrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Ornn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N* 
_output_shapes
:(: 
�
Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitchIrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2rnn_2/gradients/b_count_2*
T0*(
_output_shapes
:(:(
�
Ernn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter_grad/AddAddJrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Switch:1Vrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:(
�
Ornn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationErnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Add*
T0*
_output_shapes

:(
�
Irnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3ExitHrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Switch*
T0*
_output_shapes

:(
�
Xrnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3^rnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter`rnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^rnn_2/gradients/Sub*:
_class0
.,loc:@rnn_2/rnn/while/TensorArrayReadV3/Enter*
sourcernn_2/gradients*
_output_shapes

:: 
�
^rnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn_2/rnn/TensorArray_1*
T0*:
_class0
.,loc:@rnn_2/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
:*=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
`rnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterDrnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*:
_class0
.,loc:@rnn_2/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
: *=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
Trnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentity`rnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1Y^rnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*:
_class0
.,loc:@rnn_2/rnn/while/TensorArrayReadV3/Enter*
_output_shapes
: 
�
Zrnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Xrnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3ernn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Trnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependencyTrnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
�
Drnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Frnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterDrnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *=

frame_name/-rnn_2/gradients/rnn_2/rnn/while/while_context
�
Frnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeFrnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Lrnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
�
Ernn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchFrnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2rnn_2/gradients/b_count_2*
T0*
_output_shapes
: : 
�
Brnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddGrnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Zrnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
Lrnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationBrnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
�
Frnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitErnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 
�
=rnn_2/gradients/rnn_2/rnn/while/Switch_4_grad_1/NextIterationNextIterationVrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
{rnn_2/gradients/rnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn_2/rnn/TensorArray_1Frnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3**
_class 
loc:@rnn_2/rnn/TensorArray_1*
sourcernn_2/gradients*
_output_shapes

:: 
�
wrnn_2/gradients/rnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityFrnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3|^rnn_2/gradients/rnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0**
_class 
loc:@rnn_2/rnn/TensorArray_1*
_output_shapes
: 
�
mrnn_2/gradients/rnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3{rnn_2/gradients/rnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3"rnn_2/rnn/TensorArrayUnstack/rangewrnn_2/gradients/rnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*4
_output_shapes"
 :������������������
*
element_shape:
�
jrnn_2/gradients/rnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpn^rnn_2/gradients/rnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3G^rnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
�
rrnn_2/gradients/rnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentitymrnn_2/gradients/rnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3k^rnn_2/gradients/rnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*�
_classv
trloc:@rnn_2/gradients/rnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3*+
_output_shapes
:���������

�
trnn_2/gradients/rnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityFrnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3k^rnn_2/gradients/rnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@rnn_2/gradients/rnn_2/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes
: 
�
:rnn_2/gradients/rnn_2/rnn/transpose_grad/InvertPermutationInvertPermutationrnn_2/rnn/concat*
T0*
_output_shapes
:
�
2rnn_2/gradients/rnn_2/rnn/transpose_grad/transpose	Transposerrnn_2/gradients/rnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency:rnn_2/gradients/rnn_2/rnn/transpose_grad/InvertPermutation*
T0*+
_output_shapes
:���������
*
Tperm0
s
*rnn_2/gradients/rnn_2/Reshape_1_grad/ShapeShape	rnn_2/add*
T0*
out_type0*
_output_shapes
:
�
,rnn_2/gradients/rnn_2/Reshape_1_grad/ReshapeReshape2rnn_2/gradients/rnn_2/rnn/transpose_grad/transpose*rnn_2/gradients/rnn_2/Reshape_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

p
$rnn_2/gradients/rnn_2/add_grad/ShapeShapernn_2/MatMul*
T0*
out_type0*
_output_shapes
:
p
&rnn_2/gradients/rnn_2/add_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
�
4rnn_2/gradients/rnn_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs$rnn_2/gradients/rnn_2/add_grad/Shape&rnn_2/gradients/rnn_2/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
"rnn_2/gradients/rnn_2/add_grad/SumSum,rnn_2/gradients/rnn_2/Reshape_1_grad/Reshape4rnn_2/gradients/rnn_2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
&rnn_2/gradients/rnn_2/add_grad/ReshapeReshape"rnn_2/gradients/rnn_2/add_grad/Sum$rnn_2/gradients/rnn_2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
$rnn_2/gradients/rnn_2/add_grad/Sum_1Sum,rnn_2/gradients/rnn_2/Reshape_1_grad/Reshape6rnn_2/gradients/rnn_2/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(rnn_2/gradients/rnn_2/add_grad/Reshape_1Reshape$rnn_2/gradients/rnn_2/add_grad/Sum_1&rnn_2/gradients/rnn_2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

�
/rnn_2/gradients/rnn_2/add_grad/tuple/group_depsNoOp'^rnn_2/gradients/rnn_2/add_grad/Reshape)^rnn_2/gradients/rnn_2/add_grad/Reshape_1
�
7rnn_2/gradients/rnn_2/add_grad/tuple/control_dependencyIdentity&rnn_2/gradients/rnn_2/add_grad/Reshape0^rnn_2/gradients/rnn_2/add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@rnn_2/gradients/rnn_2/add_grad/Reshape*'
_output_shapes
:���������

�
9rnn_2/gradients/rnn_2/add_grad/tuple/control_dependency_1Identity(rnn_2/gradients/rnn_2/add_grad/Reshape_10^rnn_2/gradients/rnn_2/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@rnn_2/gradients/rnn_2/add_grad/Reshape_1*
_output_shapes
:

�
(rnn_2/gradients/rnn_2/MatMul_grad/MatMulMatMul7rnn_2/gradients/rnn_2/add_grad/tuple/control_dependencylayer_1/weights/Variable/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
*rnn_2/gradients/rnn_2/MatMul_grad/MatMul_1MatMulrnn_2/Reshape7rnn_2/gradients/rnn_2/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a(
�
2rnn_2/gradients/rnn_2/MatMul_grad/tuple/group_depsNoOp)^rnn_2/gradients/rnn_2/MatMul_grad/MatMul+^rnn_2/gradients/rnn_2/MatMul_grad/MatMul_1
�
:rnn_2/gradients/rnn_2/MatMul_grad/tuple/control_dependencyIdentity(rnn_2/gradients/rnn_2/MatMul_grad/MatMul3^rnn_2/gradients/rnn_2/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@rnn_2/gradients/rnn_2/MatMul_grad/MatMul*'
_output_shapes
:���������
�
<rnn_2/gradients/rnn_2/MatMul_grad/tuple/control_dependency_1Identity*rnn_2/gradients/rnn_2/MatMul_grad/MatMul_13^rnn_2/gradients/rnn_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@rnn_2/gradients/rnn_2/MatMul_grad/MatMul_1*
_output_shapes

:

�
rnn_2/beta1_power/initial_valueConst**
_class 
loc:@layer_1/biases/Variable*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
rnn_2/beta1_power
VariableV2*
shared_name **
_class 
loc:@layer_1/biases/Variable*
	container *
shape: *
dtype0*
_output_shapes
: 
�
rnn_2/beta1_power/AssignAssignrnn_2/beta1_powerrnn_2/beta1_power/initial_value*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn_2/beta1_power/readIdentityrnn_2/beta1_power*
T0**
_class 
loc:@layer_1/biases/Variable*
_output_shapes
: 
�
rnn_2/beta2_power/initial_valueConst**
_class 
loc:@layer_1/biases/Variable*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
rnn_2/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name **
_class 
loc:@layer_1/biases/Variable*
	container *
shape: 
�
rnn_2/beta2_power/AssignAssignrnn_2/beta2_powerrnn_2/beta2_power/initial_value*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn_2/beta2_power/readIdentityrnn_2/beta2_power*
T0**
_class 
loc:@layer_1/biases/Variable*
_output_shapes
: 
�
3rnn/layer_1/weights/Variable/Adam/Initializer/zerosConst*+
_class!
loc:@layer_1/weights/Variable*
valueB
*    *
dtype0*
_output_shapes

:

�
!rnn/layer_1/weights/Variable/Adam
VariableV2*
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *+
_class!
loc:@layer_1/weights/Variable*
	container 
�
(rnn/layer_1/weights/Variable/Adam/AssignAssign!rnn/layer_1/weights/Variable/Adam3rnn/layer_1/weights/Variable/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:

�
&rnn/layer_1/weights/Variable/Adam/readIdentity!rnn/layer_1/weights/Variable/Adam*
T0*+
_class!
loc:@layer_1/weights/Variable*
_output_shapes

:

�
5rnn/layer_1/weights/Variable/Adam_1/Initializer/zerosConst*+
_class!
loc:@layer_1/weights/Variable*
valueB
*    *
dtype0*
_output_shapes

:

�
#rnn/layer_1/weights/Variable/Adam_1
VariableV2*
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *+
_class!
loc:@layer_1/weights/Variable*
	container 
�
*rnn/layer_1/weights/Variable/Adam_1/AssignAssign#rnn/layer_1/weights/Variable/Adam_15rnn/layer_1/weights/Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:

�
(rnn/layer_1/weights/Variable/Adam_1/readIdentity#rnn/layer_1/weights/Variable/Adam_1*
T0*+
_class!
loc:@layer_1/weights/Variable*
_output_shapes

:

�
5rnn/layer_1/weights/Variable_1/Adam/Initializer/zerosConst*-
_class#
!loc:@layer_1/weights/Variable_1*
valueB
*    *
dtype0*
_output_shapes

:

�
#rnn/layer_1/weights/Variable_1/Adam
VariableV2*
shared_name *-
_class#
!loc:@layer_1/weights/Variable_1*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
*rnn/layer_1/weights/Variable_1/Adam/AssignAssign#rnn/layer_1/weights/Variable_1/Adam5rnn/layer_1/weights/Variable_1/Adam/Initializer/zeros*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
(rnn/layer_1/weights/Variable_1/Adam/readIdentity#rnn/layer_1/weights/Variable_1/Adam*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
_output_shapes

:

�
7rnn/layer_1/weights/Variable_1/Adam_1/Initializer/zerosConst*-
_class#
!loc:@layer_1/weights/Variable_1*
valueB
*    *
dtype0*
_output_shapes

:

�
%rnn/layer_1/weights/Variable_1/Adam_1
VariableV2*
shape
:
*
dtype0*
_output_shapes

:
*
shared_name *-
_class#
!loc:@layer_1/weights/Variable_1*
	container 
�
,rnn/layer_1/weights/Variable_1/Adam_1/AssignAssign%rnn/layer_1/weights/Variable_1/Adam_17rnn/layer_1/weights/Variable_1/Adam_1/Initializer/zeros*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
*rnn/layer_1/weights/Variable_1/Adam_1/readIdentity%rnn/layer_1/weights/Variable_1/Adam_1*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
_output_shapes

:

�
2rnn/layer_1/biases/Variable/Adam/Initializer/zerosConst**
_class 
loc:@layer_1/biases/Variable*
valueB
*    *
dtype0*
_output_shapes
:

�
 rnn/layer_1/biases/Variable/Adam
VariableV2*
shared_name **
_class 
loc:@layer_1/biases/Variable*
	container *
shape:
*
dtype0*
_output_shapes
:

�
'rnn/layer_1/biases/Variable/Adam/AssignAssign rnn/layer_1/biases/Variable/Adam2rnn/layer_1/biases/Variable/Adam/Initializer/zeros*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
%rnn/layer_1/biases/Variable/Adam/readIdentity rnn/layer_1/biases/Variable/Adam*
T0**
_class 
loc:@layer_1/biases/Variable*
_output_shapes
:

�
4rnn/layer_1/biases/Variable/Adam_1/Initializer/zerosConst**
_class 
loc:@layer_1/biases/Variable*
valueB
*    *
dtype0*
_output_shapes
:

�
"rnn/layer_1/biases/Variable/Adam_1
VariableV2*
shape:
*
dtype0*
_output_shapes
:
*
shared_name **
_class 
loc:@layer_1/biases/Variable*
	container 
�
)rnn/layer_1/biases/Variable/Adam_1/AssignAssign"rnn/layer_1/biases/Variable/Adam_14rnn/layer_1/biases/Variable/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:

�
'rnn/layer_1/biases/Variable/Adam_1/readIdentity"rnn/layer_1/biases/Variable/Adam_1*
T0**
_class 
loc:@layer_1/biases/Variable*
_output_shapes
:

�
4rnn/layer_1/biases/Variable_1/Adam/Initializer/zerosConst*,
_class"
 loc:@layer_1/biases/Variable_1*
valueB*    *
dtype0*
_output_shapes
:
�
"rnn/layer_1/biases/Variable_1/Adam
VariableV2*
shared_name *,
_class"
 loc:@layer_1/biases/Variable_1*
	container *
shape:*
dtype0*
_output_shapes
:
�
)rnn/layer_1/biases/Variable_1/Adam/AssignAssign"rnn/layer_1/biases/Variable_1/Adam4rnn/layer_1/biases/Variable_1/Adam/Initializer/zeros*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
'rnn/layer_1/biases/Variable_1/Adam/readIdentity"rnn/layer_1/biases/Variable_1/Adam*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
_output_shapes
:
�
6rnn/layer_1/biases/Variable_1/Adam_1/Initializer/zerosConst*,
_class"
 loc:@layer_1/biases/Variable_1*
valueB*    *
dtype0*
_output_shapes
:
�
$rnn/layer_1/biases/Variable_1/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@layer_1/biases/Variable_1*
	container 
�
+rnn/layer_1/biases/Variable_1/Adam_1/AssignAssign$rnn/layer_1/biases/Variable_1/Adam_16rnn/layer_1/biases/Variable_1/Adam_1/Initializer/zeros*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
)rnn/layer_1/biases/Variable_1/Adam_1/readIdentity$rnn/layer_1/biases/Variable_1/Adam_1*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
_output_shapes
:
]
rnn_2/Adam/learning_rateConst*
valueB
 *RI9*
dtype0*
_output_shapes
: 
U
rnn_2/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
rnn_2/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
rnn_2/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
:rnn_2/Adam/update_rnn/rnn/basic_lstm_cell/kernel/ApplyAdam	ApplyAdamrnn/rnn/basic_lstm_cell/kernel'rnn/rnn/rnn/basic_lstm_cell/kernel/Adam)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn_2/beta1_power/readrnn_2/beta2_power/readrnn_2/Adam/learning_raternn_2/Adam/beta1rnn_2/Adam/beta2rnn_2/Adam/epsilonIrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
use_locking( *
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
use_nesterov( *
_output_shapes

:(
�
8rnn_2/Adam/update_rnn/rnn/basic_lstm_cell/bias/ApplyAdam	ApplyAdamrnn/rnn/basic_lstm_cell/bias%rnn/rnn/rnn/basic_lstm_cell/bias/Adam'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1rnn_2/beta1_power/readrnn_2/beta2_power/readrnn_2/Adam/learning_raternn_2/Adam/beta1rnn_2/Adam/beta2rnn_2/Adam/epsilonJrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
use_nesterov( *
_output_shapes
:(*
use_locking( 
�
4rnn_2/Adam/update_layer_1/weights/Variable/ApplyAdam	ApplyAdamlayer_1/weights/Variable!rnn/layer_1/weights/Variable/Adam#rnn/layer_1/weights/Variable/Adam_1rnn_2/beta1_power/readrnn_2/beta2_power/readrnn_2/Adam/learning_raternn_2/Adam/beta1rnn_2/Adam/beta2rnn_2/Adam/epsilon<rnn_2/gradients/rnn_2/MatMul_grad/tuple/control_dependency_1*
T0*+
_class!
loc:@layer_1/weights/Variable*
use_nesterov( *
_output_shapes

:
*
use_locking( 
�
6rnn_2/Adam/update_layer_1/weights/Variable_1/ApplyAdam	ApplyAdamlayer_1/weights/Variable_1#rnn/layer_1/weights/Variable_1/Adam%rnn/layer_1/weights/Variable_1/Adam_1rnn_2/beta1_power/readrnn_2/beta2_power/readrnn_2/Adam/learning_raternn_2/Adam/beta1rnn_2/Adam/beta2rnn_2/Adam/epsilon>rnn_2/gradients/rnn_2/MatMul_1_grad/tuple/control_dependency_1*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
use_nesterov( *
_output_shapes

:
*
use_locking( 
�
3rnn_2/Adam/update_layer_1/biases/Variable/ApplyAdam	ApplyAdamlayer_1/biases/Variable rnn/layer_1/biases/Variable/Adam"rnn/layer_1/biases/Variable/Adam_1rnn_2/beta1_power/readrnn_2/beta2_power/readrnn_2/Adam/learning_raternn_2/Adam/beta1rnn_2/Adam/beta2rnn_2/Adam/epsilon9rnn_2/gradients/rnn_2/add_grad/tuple/control_dependency_1*
T0**
_class 
loc:@layer_1/biases/Variable*
use_nesterov( *
_output_shapes
:
*
use_locking( 
�
5rnn_2/Adam/update_layer_1/biases/Variable_1/ApplyAdam	ApplyAdamlayer_1/biases/Variable_1"rnn/layer_1/biases/Variable_1/Adam$rnn/layer_1/biases/Variable_1/Adam_1rnn_2/beta1_power/readrnn_2/beta2_power/readrnn_2/Adam/learning_raternn_2/Adam/beta1rnn_2/Adam/beta2rnn_2/Adam/epsilon;rnn_2/gradients/rnn_2/preds_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
use_nesterov( *
_output_shapes
:
�
rnn_2/Adam/mulMulrnn_2/beta1_power/readrnn_2/Adam/beta14^rnn_2/Adam/update_layer_1/biases/Variable/ApplyAdam6^rnn_2/Adam/update_layer_1/biases/Variable_1/ApplyAdam5^rnn_2/Adam/update_layer_1/weights/Variable/ApplyAdam7^rnn_2/Adam/update_layer_1/weights/Variable_1/ApplyAdam9^rnn_2/Adam/update_rnn/rnn/basic_lstm_cell/bias/ApplyAdam;^rnn_2/Adam/update_rnn/rnn/basic_lstm_cell/kernel/ApplyAdam*
T0**
_class 
loc:@layer_1/biases/Variable*
_output_shapes
: 
�
rnn_2/Adam/AssignAssignrnn_2/beta1_powerrnn_2/Adam/mul*
use_locking( *
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn_2/Adam/mul_1Mulrnn_2/beta2_power/readrnn_2/Adam/beta24^rnn_2/Adam/update_layer_1/biases/Variable/ApplyAdam6^rnn_2/Adam/update_layer_1/biases/Variable_1/ApplyAdam5^rnn_2/Adam/update_layer_1/weights/Variable/ApplyAdam7^rnn_2/Adam/update_layer_1/weights/Variable_1/ApplyAdam9^rnn_2/Adam/update_rnn/rnn/basic_lstm_cell/bias/ApplyAdam;^rnn_2/Adam/update_rnn/rnn/basic_lstm_cell/kernel/ApplyAdam*
T0**
_class 
loc:@layer_1/biases/Variable*
_output_shapes
: 
�
rnn_2/Adam/Assign_1Assignrnn_2/beta2_powerrnn_2/Adam/mul_1*
use_locking( *
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
: 
�

rnn_2/AdamNoOp^rnn_2/Adam/Assign^rnn_2/Adam/Assign_14^rnn_2/Adam/update_layer_1/biases/Variable/ApplyAdam6^rnn_2/Adam/update_layer_1/biases/Variable_1/ApplyAdam5^rnn_2/Adam/update_layer_1/weights/Variable/ApplyAdam7^rnn_2/Adam/update_layer_1/weights/Variable_1/ApplyAdam9^rnn_2/Adam/update_rnn/rnn/basic_lstm_cell/bias/ApplyAdam;^rnn_2/Adam/update_rnn/rnn/basic_lstm_cell/kernel/ApplyAdam
V
rnn_2/save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�	
rnn_2/save/SaveV2/tensor_namesConst*�
value�B�"Blayer/biases/VariableBlayer/biases/Variable_1Blayer/weights/VariableBlayer/weights/Variable_1Blayer_1/biases/VariableBlayer_1/biases/Variable_1Blayer_1/weights/VariableBlayer_1/weights/Variable_1Brnn/beta1_powerBrnn/beta2_powerBrnn/layer/biases/Variable/AdamB rnn/layer/biases/Variable/Adam_1B rnn/layer/biases/Variable_1/AdamB"rnn/layer/biases/Variable_1/Adam_1Brnn/layer/weights/Variable/AdamB!rnn/layer/weights/Variable/Adam_1B!rnn/layer/weights/Variable_1/AdamB#rnn/layer/weights/Variable_1/Adam_1B rnn/layer_1/biases/Variable/AdamB"rnn/layer_1/biases/Variable/Adam_1B"rnn/layer_1/biases/Variable_1/AdamB$rnn/layer_1/biases/Variable_1/Adam_1B!rnn/layer_1/weights/Variable/AdamB#rnn/layer_1/weights/Variable/Adam_1B#rnn/layer_1/weights/Variable_1/AdamB%rnn/layer_1/weights/Variable_1/Adam_1Brnn/rnn/basic_lstm_cell/biasBrnn/rnn/basic_lstm_cell/kernelB%rnn/rnn/rnn/basic_lstm_cell/bias/AdamB'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1B'rnn/rnn/rnn/basic_lstm_cell/kernel/AdamB)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1Brnn_2/beta1_powerBrnn_2/beta2_power*
dtype0*
_output_shapes
:"
�
"rnn_2/save/SaveV2/shape_and_slicesConst*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:"
�	
rnn_2/save/SaveV2SaveV2rnn_2/save/Constrnn_2/save/SaveV2/tensor_names"rnn_2/save/SaveV2/shape_and_sliceslayer/biases/Variablelayer/biases/Variable_1layer/weights/Variablelayer/weights/Variable_1layer_1/biases/Variablelayer_1/biases/Variable_1layer_1/weights/Variablelayer_1/weights/Variable_1rnn/beta1_powerrnn/beta2_powerrnn/layer/biases/Variable/Adam rnn/layer/biases/Variable/Adam_1 rnn/layer/biases/Variable_1/Adam"rnn/layer/biases/Variable_1/Adam_1rnn/layer/weights/Variable/Adam!rnn/layer/weights/Variable/Adam_1!rnn/layer/weights/Variable_1/Adam#rnn/layer/weights/Variable_1/Adam_1 rnn/layer_1/biases/Variable/Adam"rnn/layer_1/biases/Variable/Adam_1"rnn/layer_1/biases/Variable_1/Adam$rnn/layer_1/biases/Variable_1/Adam_1!rnn/layer_1/weights/Variable/Adam#rnn/layer_1/weights/Variable/Adam_1#rnn/layer_1/weights/Variable_1/Adam%rnn/layer_1/weights/Variable_1/Adam_1rnn/rnn/basic_lstm_cell/biasrnn/rnn/basic_lstm_cell/kernel%rnn/rnn/rnn/basic_lstm_cell/bias/Adam'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1'rnn/rnn/rnn/basic_lstm_cell/kernel/Adam)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn_2/beta1_powerrnn_2/beta2_power*0
dtypes&
$2"
�
rnn_2/save/control_dependencyIdentityrnn_2/save/Const^rnn_2/save/SaveV2*
T0*#
_class
loc:@rnn_2/save/Const*
_output_shapes
: 
�	
!rnn_2/save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�"Blayer/biases/VariableBlayer/biases/Variable_1Blayer/weights/VariableBlayer/weights/Variable_1Blayer_1/biases/VariableBlayer_1/biases/Variable_1Blayer_1/weights/VariableBlayer_1/weights/Variable_1Brnn/beta1_powerBrnn/beta2_powerBrnn/layer/biases/Variable/AdamB rnn/layer/biases/Variable/Adam_1B rnn/layer/biases/Variable_1/AdamB"rnn/layer/biases/Variable_1/Adam_1Brnn/layer/weights/Variable/AdamB!rnn/layer/weights/Variable/Adam_1B!rnn/layer/weights/Variable_1/AdamB#rnn/layer/weights/Variable_1/Adam_1B rnn/layer_1/biases/Variable/AdamB"rnn/layer_1/biases/Variable/Adam_1B"rnn/layer_1/biases/Variable_1/AdamB$rnn/layer_1/biases/Variable_1/Adam_1B!rnn/layer_1/weights/Variable/AdamB#rnn/layer_1/weights/Variable/Adam_1B#rnn/layer_1/weights/Variable_1/AdamB%rnn/layer_1/weights/Variable_1/Adam_1Brnn/rnn/basic_lstm_cell/biasBrnn/rnn/basic_lstm_cell/kernelB%rnn/rnn/rnn/basic_lstm_cell/bias/AdamB'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1B'rnn/rnn/rnn/basic_lstm_cell/kernel/AdamB)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1Brnn_2/beta1_powerBrnn_2/beta2_power*
dtype0*
_output_shapes
:"
�
%rnn_2/save/RestoreV2/shape_and_slicesConst"/device:CPU:0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:"
�
rnn_2/save/RestoreV2	RestoreV2rnn_2/save/Const!rnn_2/save/RestoreV2/tensor_names%rnn_2/save/RestoreV2/shape_and_slices"/device:CPU:0*0
dtypes&
$2"*�
_output_shapes�
�::::::::::::::::::::::::::::::::::
�
rnn_2/save/AssignAssignlayer/biases/Variablernn_2/save/RestoreV2*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn_2/save/Assign_1Assignlayer/biases/Variable_1rnn_2/save/RestoreV2:1*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_2/save/Assign_2Assignlayer/weights/Variablernn_2/save/RestoreV2:2*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_2/save/Assign_3Assignlayer/weights/Variable_1rnn_2/save/RestoreV2:3*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_2/save/Assign_4Assignlayer_1/biases/Variablernn_2/save/RestoreV2:4*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn_2/save/Assign_5Assignlayer_1/biases/Variable_1rnn_2/save/RestoreV2:5*
use_locking(*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_2/save/Assign_6Assignlayer_1/weights/Variablernn_2/save/RestoreV2:6*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_2/save/Assign_7Assignlayer_1/weights/Variable_1rnn_2/save/RestoreV2:7*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_2/save/Assign_8Assignrnn/beta1_powerrnn_2/save/RestoreV2:8*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn_2/save/Assign_9Assignrnn/beta2_powerrnn_2/save/RestoreV2:9*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(
�
rnn_2/save/Assign_10Assignrnn/layer/biases/Variable/Adamrnn_2/save/RestoreV2:10*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_2/save/Assign_11Assign rnn/layer/biases/Variable/Adam_1rnn_2/save/RestoreV2:11*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_2/save/Assign_12Assign rnn/layer/biases/Variable_1/Adamrnn_2/save/RestoreV2:12*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_2/save/Assign_13Assign"rnn/layer/biases/Variable_1/Adam_1rnn_2/save/RestoreV2:13*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_2/save/Assign_14Assignrnn/layer/weights/Variable/Adamrnn_2/save/RestoreV2:14*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_2/save/Assign_15Assign!rnn/layer/weights/Variable/Adam_1rnn_2/save/RestoreV2:15*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_2/save/Assign_16Assign!rnn/layer/weights/Variable_1/Adamrnn_2/save/RestoreV2:16*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_2/save/Assign_17Assign#rnn/layer/weights/Variable_1/Adam_1rnn_2/save/RestoreV2:17*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_2/save/Assign_18Assign rnn/layer_1/biases/Variable/Adamrnn_2/save/RestoreV2:18*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn_2/save/Assign_19Assign"rnn/layer_1/biases/Variable/Adam_1rnn_2/save/RestoreV2:19*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_2/save/Assign_20Assign"rnn/layer_1/biases/Variable_1/Adamrnn_2/save/RestoreV2:20*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_2/save/Assign_21Assign$rnn/layer_1/biases/Variable_1/Adam_1rnn_2/save/RestoreV2:21*
use_locking(*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_2/save/Assign_22Assign!rnn/layer_1/weights/Variable/Adamrnn_2/save/RestoreV2:22*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_2/save/Assign_23Assign#rnn/layer_1/weights/Variable/Adam_1rnn_2/save/RestoreV2:23*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_2/save/Assign_24Assign#rnn/layer_1/weights/Variable_1/Adamrnn_2/save/RestoreV2:24*
use_locking(*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_2/save/Assign_25Assign%rnn/layer_1/weights/Variable_1/Adam_1rnn_2/save/RestoreV2:25*
use_locking(*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_2/save/Assign_26Assignrnn/rnn/basic_lstm_cell/biasrnn_2/save/RestoreV2:26*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
rnn_2/save/Assign_27Assignrnn/rnn/basic_lstm_cell/kernelrnn_2/save/RestoreV2:27*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
rnn_2/save/Assign_28Assign%rnn/rnn/rnn/basic_lstm_cell/bias/Adamrnn_2/save/RestoreV2:28*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
rnn_2/save/Assign_29Assign'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1rnn_2/save/RestoreV2:29*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
rnn_2/save/Assign_30Assign'rnn/rnn/rnn/basic_lstm_cell/kernel/Adamrnn_2/save/RestoreV2:30*
use_locking(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
rnn_2/save/Assign_31Assign)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn_2/save/RestoreV2:31*
use_locking(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
rnn_2/save/Assign_32Assignrnn_2/beta1_powerrnn_2/save/RestoreV2:32*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn_2/save/Assign_33Assignrnn_2/beta2_powerrnn_2/save/RestoreV2:33*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(
�
rnn_2/save/restore_allNoOp^rnn_2/save/Assign^rnn_2/save/Assign_1^rnn_2/save/Assign_10^rnn_2/save/Assign_11^rnn_2/save/Assign_12^rnn_2/save/Assign_13^rnn_2/save/Assign_14^rnn_2/save/Assign_15^rnn_2/save/Assign_16^rnn_2/save/Assign_17^rnn_2/save/Assign_18^rnn_2/save/Assign_19^rnn_2/save/Assign_2^rnn_2/save/Assign_20^rnn_2/save/Assign_21^rnn_2/save/Assign_22^rnn_2/save/Assign_23^rnn_2/save/Assign_24^rnn_2/save/Assign_25^rnn_2/save/Assign_26^rnn_2/save/Assign_27^rnn_2/save/Assign_28^rnn_2/save/Assign_29^rnn_2/save/Assign_3^rnn_2/save/Assign_30^rnn_2/save/Assign_31^rnn_2/save/Assign_32^rnn_2/save/Assign_33^rnn_2/save/Assign_4^rnn_2/save/Assign_5^rnn_2/save/Assign_6^rnn_2/save/Assign_7^rnn_2/save/Assign_8^rnn_2/save/Assign_9
V
rnn_2/PlaceholderPlaceholder*
shape:*
dtype0*
_output_shapes
:
�


rnn_2/initNoOp^layer/biases/Variable/Assign^layer/biases/Variable_1/Assign^layer/weights/Variable/Assign ^layer/weights/Variable_1/Assign^layer_1/biases/Variable/Assign!^layer_1/biases/Variable_1/Assign ^layer_1/weights/Variable/Assign"^layer_1/weights/Variable_1/Assign^rnn/beta1_power/Assign^rnn/beta2_power/Assign&^rnn/layer/biases/Variable/Adam/Assign(^rnn/layer/biases/Variable/Adam_1/Assign(^rnn/layer/biases/Variable_1/Adam/Assign*^rnn/layer/biases/Variable_1/Adam_1/Assign'^rnn/layer/weights/Variable/Adam/Assign)^rnn/layer/weights/Variable/Adam_1/Assign)^rnn/layer/weights/Variable_1/Adam/Assign+^rnn/layer/weights/Variable_1/Adam_1/Assign(^rnn/layer_1/biases/Variable/Adam/Assign*^rnn/layer_1/biases/Variable/Adam_1/Assign*^rnn/layer_1/biases/Variable_1/Adam/Assign,^rnn/layer_1/biases/Variable_1/Adam_1/Assign)^rnn/layer_1/weights/Variable/Adam/Assign+^rnn/layer_1/weights/Variable/Adam_1/Assign+^rnn/layer_1/weights/Variable_1/Adam/Assign-^rnn/layer_1/weights/Variable_1/Adam_1/Assign$^rnn/rnn/basic_lstm_cell/bias/Assign&^rnn/rnn/basic_lstm_cell/kernel/Assign-^rnn/rnn/rnn/basic_lstm_cell/bias/Adam/Assign/^rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1/Assign/^rnn/rnn/rnn/basic_lstm_cell/kernel/Adam/Assign1^rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1/Assign^rnn_2/beta1_power/Assign^rnn_2/beta2_power/Assign
X
rnn_2/save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
 rnn_2/save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_4c713ea2f47e429b8dd8196ff471b37a/part*
dtype0*
_output_shapes
: 
�
rnn_2/save_1/StringJoin
StringJoinrnn_2/save_1/Const rnn_2/save_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Y
rnn_2/save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
s
"rnn_2/save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_2/save_1/ShardedFilenameShardedFilenamernn_2/save_1/StringJoin"rnn_2/save_1/ShardedFilename/shardrnn_2/save_1/num_shards"/device:CPU:0*
_output_shapes
: 
�	
 rnn_2/save_1/SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�"Blayer/biases/VariableBlayer/biases/Variable_1Blayer/weights/VariableBlayer/weights/Variable_1Blayer_1/biases/VariableBlayer_1/biases/Variable_1Blayer_1/weights/VariableBlayer_1/weights/Variable_1Brnn/beta1_powerBrnn/beta2_powerBrnn/layer/biases/Variable/AdamB rnn/layer/biases/Variable/Adam_1B rnn/layer/biases/Variable_1/AdamB"rnn/layer/biases/Variable_1/Adam_1Brnn/layer/weights/Variable/AdamB!rnn/layer/weights/Variable/Adam_1B!rnn/layer/weights/Variable_1/AdamB#rnn/layer/weights/Variable_1/Adam_1B rnn/layer_1/biases/Variable/AdamB"rnn/layer_1/biases/Variable/Adam_1B"rnn/layer_1/biases/Variable_1/AdamB$rnn/layer_1/biases/Variable_1/Adam_1B!rnn/layer_1/weights/Variable/AdamB#rnn/layer_1/weights/Variable/Adam_1B#rnn/layer_1/weights/Variable_1/AdamB%rnn/layer_1/weights/Variable_1/Adam_1Brnn/rnn/basic_lstm_cell/biasBrnn/rnn/basic_lstm_cell/kernelB%rnn/rnn/rnn/basic_lstm_cell/bias/AdamB'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1B'rnn/rnn/rnn/basic_lstm_cell/kernel/AdamB)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1Brnn_2/beta1_powerBrnn_2/beta2_power*
dtype0*
_output_shapes
:"
�
$rnn_2/save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:"
�	
rnn_2/save_1/SaveV2SaveV2rnn_2/save_1/ShardedFilename rnn_2/save_1/SaveV2/tensor_names$rnn_2/save_1/SaveV2/shape_and_sliceslayer/biases/Variablelayer/biases/Variable_1layer/weights/Variablelayer/weights/Variable_1layer_1/biases/Variablelayer_1/biases/Variable_1layer_1/weights/Variablelayer_1/weights/Variable_1rnn/beta1_powerrnn/beta2_powerrnn/layer/biases/Variable/Adam rnn/layer/biases/Variable/Adam_1 rnn/layer/biases/Variable_1/Adam"rnn/layer/biases/Variable_1/Adam_1rnn/layer/weights/Variable/Adam!rnn/layer/weights/Variable/Adam_1!rnn/layer/weights/Variable_1/Adam#rnn/layer/weights/Variable_1/Adam_1 rnn/layer_1/biases/Variable/Adam"rnn/layer_1/biases/Variable/Adam_1"rnn/layer_1/biases/Variable_1/Adam$rnn/layer_1/biases/Variable_1/Adam_1!rnn/layer_1/weights/Variable/Adam#rnn/layer_1/weights/Variable/Adam_1#rnn/layer_1/weights/Variable_1/Adam%rnn/layer_1/weights/Variable_1/Adam_1rnn/rnn/basic_lstm_cell/biasrnn/rnn/basic_lstm_cell/kernel%rnn/rnn/rnn/basic_lstm_cell/bias/Adam'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1'rnn/rnn/rnn/basic_lstm_cell/kernel/Adam)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn_2/beta1_powerrnn_2/beta2_power"/device:CPU:0*0
dtypes&
$2"
�
rnn_2/save_1/control_dependencyIdentityrnn_2/save_1/ShardedFilename^rnn_2/save_1/SaveV2"/device:CPU:0*
T0*/
_class%
#!loc:@rnn_2/save_1/ShardedFilename*
_output_shapes
: 
�
3rnn_2/save_1/MergeV2Checkpoints/checkpoint_prefixesPackrnn_2/save_1/ShardedFilename ^rnn_2/save_1/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
rnn_2/save_1/MergeV2CheckpointsMergeV2Checkpoints3rnn_2/save_1/MergeV2Checkpoints/checkpoint_prefixesrnn_2/save_1/Const"/device:CPU:0*
delete_old_dirs(
�
rnn_2/save_1/IdentityIdentityrnn_2/save_1/Const ^rnn_2/save_1/MergeV2Checkpoints ^rnn_2/save_1/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
�	
#rnn_2/save_1/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�"Blayer/biases/VariableBlayer/biases/Variable_1Blayer/weights/VariableBlayer/weights/Variable_1Blayer_1/biases/VariableBlayer_1/biases/Variable_1Blayer_1/weights/VariableBlayer_1/weights/Variable_1Brnn/beta1_powerBrnn/beta2_powerBrnn/layer/biases/Variable/AdamB rnn/layer/biases/Variable/Adam_1B rnn/layer/biases/Variable_1/AdamB"rnn/layer/biases/Variable_1/Adam_1Brnn/layer/weights/Variable/AdamB!rnn/layer/weights/Variable/Adam_1B!rnn/layer/weights/Variable_1/AdamB#rnn/layer/weights/Variable_1/Adam_1B rnn/layer_1/biases/Variable/AdamB"rnn/layer_1/biases/Variable/Adam_1B"rnn/layer_1/biases/Variable_1/AdamB$rnn/layer_1/biases/Variable_1/Adam_1B!rnn/layer_1/weights/Variable/AdamB#rnn/layer_1/weights/Variable/Adam_1B#rnn/layer_1/weights/Variable_1/AdamB%rnn/layer_1/weights/Variable_1/Adam_1Brnn/rnn/basic_lstm_cell/biasBrnn/rnn/basic_lstm_cell/kernelB%rnn/rnn/rnn/basic_lstm_cell/bias/AdamB'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1B'rnn/rnn/rnn/basic_lstm_cell/kernel/AdamB)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1Brnn_2/beta1_powerBrnn_2/beta2_power*
dtype0*
_output_shapes
:"
�
'rnn_2/save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:"
�
rnn_2/save_1/RestoreV2	RestoreV2rnn_2/save_1/Const#rnn_2/save_1/RestoreV2/tensor_names'rnn_2/save_1/RestoreV2/shape_and_slices"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::*0
dtypes&
$2"
�
rnn_2/save_1/AssignAssignlayer/biases/Variablernn_2/save_1/RestoreV2*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn_2/save_1/Assign_1Assignlayer/biases/Variable_1rnn_2/save_1/RestoreV2:1*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_2/save_1/Assign_2Assignlayer/weights/Variablernn_2/save_1/RestoreV2:2*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_2/save_1/Assign_3Assignlayer/weights/Variable_1rnn_2/save_1/RestoreV2:3*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_2/save_1/Assign_4Assignlayer_1/biases/Variablernn_2/save_1/RestoreV2:4*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn_2/save_1/Assign_5Assignlayer_1/biases/Variable_1rnn_2/save_1/RestoreV2:5*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_2/save_1/Assign_6Assignlayer_1/weights/Variablernn_2/save_1/RestoreV2:6*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_2/save_1/Assign_7Assignlayer_1/weights/Variable_1rnn_2/save_1/RestoreV2:7*
use_locking(*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_2/save_1/Assign_8Assignrnn/beta1_powerrnn_2/save_1/RestoreV2:8*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(
�
rnn_2/save_1/Assign_9Assignrnn/beta2_powerrnn_2/save_1/RestoreV2:9*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(
�
rnn_2/save_1/Assign_10Assignrnn/layer/biases/Variable/Adamrnn_2/save_1/RestoreV2:10*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn_2/save_1/Assign_11Assign rnn/layer/biases/Variable/Adam_1rnn_2/save_1/RestoreV2:11*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn_2/save_1/Assign_12Assign rnn/layer/biases/Variable_1/Adamrnn_2/save_1/RestoreV2:12*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_2/save_1/Assign_13Assign"rnn/layer/biases/Variable_1/Adam_1rnn_2/save_1/RestoreV2:13*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_2/save_1/Assign_14Assignrnn/layer/weights/Variable/Adamrnn_2/save_1/RestoreV2:14*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_2/save_1/Assign_15Assign!rnn/layer/weights/Variable/Adam_1rnn_2/save_1/RestoreV2:15*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_2/save_1/Assign_16Assign!rnn/layer/weights/Variable_1/Adamrnn_2/save_1/RestoreV2:16*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_2/save_1/Assign_17Assign#rnn/layer/weights/Variable_1/Adam_1rnn_2/save_1/RestoreV2:17*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_2/save_1/Assign_18Assign rnn/layer_1/biases/Variable/Adamrnn_2/save_1/RestoreV2:18*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_2/save_1/Assign_19Assign"rnn/layer_1/biases/Variable/Adam_1rnn_2/save_1/RestoreV2:19*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_2/save_1/Assign_20Assign"rnn/layer_1/biases/Variable_1/Adamrnn_2/save_1/RestoreV2:20*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_2/save_1/Assign_21Assign$rnn/layer_1/biases/Variable_1/Adam_1rnn_2/save_1/RestoreV2:21*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_2/save_1/Assign_22Assign!rnn/layer_1/weights/Variable/Adamrnn_2/save_1/RestoreV2:22*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_2/save_1/Assign_23Assign#rnn/layer_1/weights/Variable/Adam_1rnn_2/save_1/RestoreV2:23*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_2/save_1/Assign_24Assign#rnn/layer_1/weights/Variable_1/Adamrnn_2/save_1/RestoreV2:24*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_2/save_1/Assign_25Assign%rnn/layer_1/weights/Variable_1/Adam_1rnn_2/save_1/RestoreV2:25*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_2/save_1/Assign_26Assignrnn/rnn/basic_lstm_cell/biasrnn_2/save_1/RestoreV2:26*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
rnn_2/save_1/Assign_27Assignrnn/rnn/basic_lstm_cell/kernelrnn_2/save_1/RestoreV2:27*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
rnn_2/save_1/Assign_28Assign%rnn/rnn/rnn/basic_lstm_cell/bias/Adamrnn_2/save_1/RestoreV2:28*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
rnn_2/save_1/Assign_29Assign'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1rnn_2/save_1/RestoreV2:29*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
rnn_2/save_1/Assign_30Assign'rnn/rnn/rnn/basic_lstm_cell/kernel/Adamrnn_2/save_1/RestoreV2:30*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
rnn_2/save_1/Assign_31Assign)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn_2/save_1/RestoreV2:31*
use_locking(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
rnn_2/save_1/Assign_32Assignrnn_2/beta1_powerrnn_2/save_1/RestoreV2:32*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn_2/save_1/Assign_33Assignrnn_2/beta2_powerrnn_2/save_1/RestoreV2:33*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn_2/save_1/restore_shardNoOp^rnn_2/save_1/Assign^rnn_2/save_1/Assign_1^rnn_2/save_1/Assign_10^rnn_2/save_1/Assign_11^rnn_2/save_1/Assign_12^rnn_2/save_1/Assign_13^rnn_2/save_1/Assign_14^rnn_2/save_1/Assign_15^rnn_2/save_1/Assign_16^rnn_2/save_1/Assign_17^rnn_2/save_1/Assign_18^rnn_2/save_1/Assign_19^rnn_2/save_1/Assign_2^rnn_2/save_1/Assign_20^rnn_2/save_1/Assign_21^rnn_2/save_1/Assign_22^rnn_2/save_1/Assign_23^rnn_2/save_1/Assign_24^rnn_2/save_1/Assign_25^rnn_2/save_1/Assign_26^rnn_2/save_1/Assign_27^rnn_2/save_1/Assign_28^rnn_2/save_1/Assign_29^rnn_2/save_1/Assign_3^rnn_2/save_1/Assign_30^rnn_2/save_1/Assign_31^rnn_2/save_1/Assign_32^rnn_2/save_1/Assign_33^rnn_2/save_1/Assign_4^rnn_2/save_1/Assign_5^rnn_2/save_1/Assign_6^rnn_2/save_1/Assign_7^rnn_2/save_1/Assign_8^rnn_2/save_1/Assign_9
=
rnn_2/save_1/restore_allNoOp^rnn_2/save_1/restore_shard
d
rnn_3/Reshape/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
w
rnn_3/ReshapeReshapeinputs_1rnn_3/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:���������
�
rnn_3/MatMulMatMulrnn_3/Reshapelayer_1/weights/Variable/read*
transpose_b( *
T0*'
_output_shapes
:���������
*
transpose_a( 
n
	rnn_3/addAddrnn_3/MatMullayer_1/biases/Variable/read*
T0*'
_output_shapes
:���������

j
rnn_3/Reshape_1/shapeConst*!
valueB"����   
   *
dtype0*
_output_shapes
:
�
rnn_3/Reshape_1Reshape	rnn_3/addrnn_3/Reshape_1/shape*
T0*
Tshape0*+
_output_shapes
:���������

l
"rnn_3/BasicLSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
n
$rnn_3/BasicLSTMCellZeroState/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
j
(rnn_3/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
#rnn_3/BasicLSTMCellZeroState/concatConcatV2"rnn_3/BasicLSTMCellZeroState/Const$rnn_3/BasicLSTMCellZeroState/Const_1(rnn_3/BasicLSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
m
(rnn_3/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"rnn_3/BasicLSTMCellZeroState/zerosFill#rnn_3/BasicLSTMCellZeroState/concat(rnn_3/BasicLSTMCellZeroState/zeros/Const*
T0*

index_type0*
_output_shapes

:

n
$rnn_3/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
n
$rnn_3/BasicLSTMCellZeroState/Const_3Const*
valueB:
*
dtype0*
_output_shapes
:
n
$rnn_3/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
n
$rnn_3/BasicLSTMCellZeroState/Const_5Const*
valueB:
*
dtype0*
_output_shapes
:
l
*rnn_3/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
%rnn_3/BasicLSTMCellZeroState/concat_1ConcatV2$rnn_3/BasicLSTMCellZeroState/Const_4$rnn_3/BasicLSTMCellZeroState/Const_5*rnn_3/BasicLSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
o
*rnn_3/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
$rnn_3/BasicLSTMCellZeroState/zeros_1Fill%rnn_3/BasicLSTMCellZeroState/concat_1*rnn_3/BasicLSTMCellZeroState/zeros_1/Const*
T0*

index_type0*
_output_shapes

:

n
$rnn_3/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
n
$rnn_3/BasicLSTMCellZeroState/Const_7Const*
valueB:
*
dtype0*
_output_shapes
:
P
rnn_3/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
W
rnn_3/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
W
rnn_3/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
~
rnn_3/rnn/rangeRangernn_3/rnn/range/startrnn_3/rnn/Rankrnn_3/rnn/range/delta*

Tidx0*
_output_shapes
:
j
rnn_3/rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
W
rnn_3/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_3/rnn/concatConcatV2rnn_3/rnn/concat/values_0rnn_3/rnn/rangernn_3/rnn/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
rnn_3/rnn/transpose	Transposernn_3/Reshape_1rnn_3/rnn/concat*
Tperm0*
T0*+
_output_shapes
:���������

b
rnn_3/rnn/ShapeShapernn_3/rnn/transpose*
T0*
out_type0*
_output_shapes
:
g
rnn_3/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
rnn_3/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
i
rnn_3/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
rnn_3/rnn/strided_sliceStridedSlicernn_3/rnn/Shapernn_3/rnn/strided_slice/stackrnn_3/rnn/strided_slice/stack_1rnn_3/rnn/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
d
rnn_3/rnn/Shape_1Shapernn_3/rnn/transpose*
T0*
out_type0*
_output_shapes
:
i
rnn_3/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
k
!rnn_3/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
k
!rnn_3/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
rnn_3/rnn/strided_slice_1StridedSlicernn_3/rnn/Shape_1rnn_3/rnn/strided_slice_1/stack!rnn_3/rnn/strided_slice_1/stack_1!rnn_3/rnn/strided_slice_1/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
d
rnn_3/rnn/Shape_2Shapernn_3/rnn/transpose*
T0*
out_type0*
_output_shapes
:
i
rnn_3/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
k
!rnn_3/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
k
!rnn_3/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
rnn_3/rnn/strided_slice_2StridedSlicernn_3/rnn/Shape_2rnn_3/rnn/strided_slice_2/stack!rnn_3/rnn/strided_slice_2/stack_1!rnn_3/rnn/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Z
rnn_3/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_3/rnn/ExpandDims
ExpandDimsrnn_3/rnn/strided_slice_2rnn_3/rnn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
Y
rnn_3/rnn/ConstConst*
valueB:
*
dtype0*
_output_shapes
:
Y
rnn_3/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_3/rnn/concat_1ConcatV2rnn_3/rnn/ExpandDimsrnn_3/rnn/Constrnn_3/rnn/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
Z
rnn_3/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
rnn_3/rnn/zerosFillrnn_3/rnn/concat_1rnn_3/rnn/zeros/Const*
T0*

index_type0*'
_output_shapes
:���������

P
rnn_3/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_3/rnn/TensorArrayTensorArrayV3rnn_3/rnn/strided_slice_1*5
tensor_array_name rnn_3/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *$
element_shape:���������
*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
�
rnn_3/rnn/TensorArray_1TensorArrayV3rnn_3/rnn/strided_slice_1*4
tensor_array_namernn_3/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *$
element_shape:���������
*
clear_after_read(*
dynamic_size( *
identical_element_shapes(
u
"rnn_3/rnn/TensorArrayUnstack/ShapeShapernn_3/rnn/transpose*
T0*
out_type0*
_output_shapes
:
z
0rnn_3/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2rnn_3/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2rnn_3/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
*rnn_3/rnn/TensorArrayUnstack/strided_sliceStridedSlice"rnn_3/rnn/TensorArrayUnstack/Shape0rnn_3/rnn/TensorArrayUnstack/strided_slice/stack2rnn_3/rnn/TensorArrayUnstack/strided_slice/stack_12rnn_3/rnn/TensorArrayUnstack/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
j
(rnn_3/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
j
(rnn_3/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
"rnn_3/rnn/TensorArrayUnstack/rangeRange(rnn_3/rnn/TensorArrayUnstack/range/start*rnn_3/rnn/TensorArrayUnstack/strided_slice(rnn_3/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:���������*

Tidx0
�
Drnn_3/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn_3/rnn/TensorArray_1"rnn_3/rnn/TensorArrayUnstack/rangernn_3/rnn/transposernn_3/rnn/TensorArray_1:1*
T0*&
_class
loc:@rnn_3/rnn/transpose*
_output_shapes
: 
U
rnn_3/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
m
rnn_3/rnn/MaximumMaximumrnn_3/rnn/Maximum/xrnn_3/rnn/strided_slice_1*
T0*
_output_shapes
: 
k
rnn_3/rnn/MinimumMinimumrnn_3/rnn/strided_slice_1rnn_3/rnn/Maximum*
T0*
_output_shapes
: 
c
!rnn_3/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_3/rnn/while/EnterEnter!rnn_3/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *-

frame_namernn_3/rnn/while/while_context
�
rnn_3/rnn/while/Enter_1Enterrnn_3/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *-

frame_namernn_3/rnn/while/while_context
�
rnn_3/rnn/while/Enter_2Enterrnn_3/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *-

frame_namernn_3/rnn/while/while_context
�
rnn_3/rnn/while/Enter_3Enter"rnn_3/BasicLSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*-

frame_namernn_3/rnn/while/while_context
�
rnn_3/rnn/while/Enter_4Enter$rnn_3/BasicLSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*-

frame_namernn_3/rnn/while/while_context
�
rnn_3/rnn/while/MergeMergernn_3/rnn/while/Enterrnn_3/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 
�
rnn_3/rnn/while/Merge_1Mergernn_3/rnn/while/Enter_1rnn_3/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
rnn_3/rnn/while/Merge_2Mergernn_3/rnn/while/Enter_2rnn_3/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
�
rnn_3/rnn/while/Merge_3Mergernn_3/rnn/while/Enter_3rnn_3/rnn/while/NextIteration_3*
T0*
N* 
_output_shapes
:
: 
�
rnn_3/rnn/while/Merge_4Mergernn_3/rnn/while/Enter_4rnn_3/rnn/while/NextIteration_4*
T0*
N* 
_output_shapes
:
: 
p
rnn_3/rnn/while/LessLessrnn_3/rnn/while/Mergernn_3/rnn/while/Less/Enter*
T0*
_output_shapes
: 
�
rnn_3/rnn/while/Less/EnterEnterrnn_3/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *-

frame_namernn_3/rnn/while/while_context
v
rnn_3/rnn/while/Less_1Lessrnn_3/rnn/while/Merge_1rnn_3/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
�
rnn_3/rnn/while/Less_1/EnterEnterrnn_3/rnn/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *-

frame_namernn_3/rnn/while/while_context
n
rnn_3/rnn/while/LogicalAnd
LogicalAndrnn_3/rnn/while/Lessrnn_3/rnn/while/Less_1*
_output_shapes
: 
X
rnn_3/rnn/while/LoopCondLoopCondrnn_3/rnn/while/LogicalAnd*
_output_shapes
: 
�
rnn_3/rnn/while/SwitchSwitchrnn_3/rnn/while/Mergernn_3/rnn/while/LoopCond*
T0*(
_class
loc:@rnn_3/rnn/while/Merge*
_output_shapes
: : 
�
rnn_3/rnn/while/Switch_1Switchrnn_3/rnn/while/Merge_1rnn_3/rnn/while/LoopCond*
T0**
_class 
loc:@rnn_3/rnn/while/Merge_1*
_output_shapes
: : 
�
rnn_3/rnn/while/Switch_2Switchrnn_3/rnn/while/Merge_2rnn_3/rnn/while/LoopCond*
T0**
_class 
loc:@rnn_3/rnn/while/Merge_2*
_output_shapes
: : 
�
rnn_3/rnn/while/Switch_3Switchrnn_3/rnn/while/Merge_3rnn_3/rnn/while/LoopCond*
T0**
_class 
loc:@rnn_3/rnn/while/Merge_3*(
_output_shapes
:
:

�
rnn_3/rnn/while/Switch_4Switchrnn_3/rnn/while/Merge_4rnn_3/rnn/while/LoopCond*
T0**
_class 
loc:@rnn_3/rnn/while/Merge_4*(
_output_shapes
:
:

_
rnn_3/rnn/while/IdentityIdentityrnn_3/rnn/while/Switch:1*
T0*
_output_shapes
: 
c
rnn_3/rnn/while/Identity_1Identityrnn_3/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
c
rnn_3/rnn/while/Identity_2Identityrnn_3/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
k
rnn_3/rnn/while/Identity_3Identityrnn_3/rnn/while/Switch_3:1*
T0*
_output_shapes

:

k
rnn_3/rnn/while/Identity_4Identityrnn_3/rnn/while/Switch_4:1*
T0*
_output_shapes

:

r
rnn_3/rnn/while/add/yConst^rnn_3/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
l
rnn_3/rnn/while/addAddrnn_3/rnn/while/Identityrnn_3/rnn/while/add/y*
T0*
_output_shapes
: 
�
!rnn_3/rnn/while/TensorArrayReadV3TensorArrayReadV3'rnn_3/rnn/while/TensorArrayReadV3/Enterrnn_3/rnn/while/Identity_1)rnn_3/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:���������

�
'rnn_3/rnn/while/TensorArrayReadV3/EnterEnterrnn_3/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_3/rnn/while/while_context
�
)rnn_3/rnn/while/TensorArrayReadV3/Enter_1EnterDrnn_3/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *-

frame_namernn_3/rnn/while/while_context
�
%rnn_3/rnn/while/basic_lstm_cell/ConstConst^rnn_3/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
+rnn_3/rnn/while/basic_lstm_cell/concat/axisConst^rnn_3/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
&rnn_3/rnn/while/basic_lstm_cell/concatConcatV2!rnn_3/rnn/while/TensorArrayReadV3rnn_3/rnn/while/Identity_4+rnn_3/rnn/while/basic_lstm_cell/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
�
&rnn_3/rnn/while/basic_lstm_cell/MatMulMatMul&rnn_3/rnn/while/basic_lstm_cell/concat,rnn_3/rnn/while/basic_lstm_cell/MatMul/Enter*
transpose_b( *
T0*
_output_shapes

:(*
transpose_a( 
�
,rnn_3/rnn/while/basic_lstm_cell/MatMul/EnterEnter#rnn/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*-

frame_namernn_3/rnn/while/while_context
�
'rnn_3/rnn/while/basic_lstm_cell/BiasAddBiasAdd&rnn_3/rnn/while/basic_lstm_cell/MatMul-rnn_3/rnn/while/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes

:(
�
-rnn_3/rnn/while/basic_lstm_cell/BiasAdd/EnterEnter!rnn/rnn/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:(*-

frame_namernn_3/rnn/while/while_context
�
'rnn_3/rnn/while/basic_lstm_cell/Const_1Const^rnn_3/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
%rnn_3/rnn/while/basic_lstm_cell/splitSplit%rnn_3/rnn/while/basic_lstm_cell/Const'rnn_3/rnn/while/basic_lstm_cell/BiasAdd*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split
�
'rnn_3/rnn/while/basic_lstm_cell/Const_2Const^rnn_3/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
#rnn_3/rnn/while/basic_lstm_cell/AddAdd'rnn_3/rnn/while/basic_lstm_cell/split:2'rnn_3/rnn/while/basic_lstm_cell/Const_2*
T0*
_output_shapes

:

�
'rnn_3/rnn/while/basic_lstm_cell/SigmoidSigmoid#rnn_3/rnn/while/basic_lstm_cell/Add*
T0*
_output_shapes

:

�
#rnn_3/rnn/while/basic_lstm_cell/MulMulrnn_3/rnn/while/Identity_3'rnn_3/rnn/while/basic_lstm_cell/Sigmoid*
T0*
_output_shapes

:

�
)rnn_3/rnn/while/basic_lstm_cell/Sigmoid_1Sigmoid%rnn_3/rnn/while/basic_lstm_cell/split*
T0*
_output_shapes

:

~
$rnn_3/rnn/while/basic_lstm_cell/TanhTanh'rnn_3/rnn/while/basic_lstm_cell/split:1*
T0*
_output_shapes

:

�
%rnn_3/rnn/while/basic_lstm_cell/Mul_1Mul)rnn_3/rnn/while/basic_lstm_cell/Sigmoid_1$rnn_3/rnn/while/basic_lstm_cell/Tanh*
T0*
_output_shapes

:

�
%rnn_3/rnn/while/basic_lstm_cell/Add_1Add#rnn_3/rnn/while/basic_lstm_cell/Mul%rnn_3/rnn/while/basic_lstm_cell/Mul_1*
T0*
_output_shapes

:

~
&rnn_3/rnn/while/basic_lstm_cell/Tanh_1Tanh%rnn_3/rnn/while/basic_lstm_cell/Add_1*
T0*
_output_shapes

:

�
)rnn_3/rnn/while/basic_lstm_cell/Sigmoid_2Sigmoid'rnn_3/rnn/while/basic_lstm_cell/split:3*
T0*
_output_shapes

:

�
%rnn_3/rnn/while/basic_lstm_cell/Mul_2Mul&rnn_3/rnn/while/basic_lstm_cell/Tanh_1)rnn_3/rnn/while/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes

:

�
3rnn_3/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV39rnn_3/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn_3/rnn/while/Identity_1%rnn_3/rnn/while/basic_lstm_cell/Mul_2rnn_3/rnn/while/Identity_2*
T0*8
_class.
,*loc:@rnn_3/rnn/while/basic_lstm_cell/Mul_2*
_output_shapes
: 
�
9rnn_3/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn_3/rnn/TensorArray*
is_constant(*
_output_shapes
:*-

frame_namernn_3/rnn/while/while_context*
T0*8
_class.
,*loc:@rnn_3/rnn/while/basic_lstm_cell/Mul_2*
parallel_iterations 
t
rnn_3/rnn/while/add_1/yConst^rnn_3/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
r
rnn_3/rnn/while/add_1Addrnn_3/rnn/while/Identity_1rnn_3/rnn/while/add_1/y*
T0*
_output_shapes
: 
d
rnn_3/rnn/while/NextIterationNextIterationrnn_3/rnn/while/add*
T0*
_output_shapes
: 
h
rnn_3/rnn/while/NextIteration_1NextIterationrnn_3/rnn/while/add_1*
T0*
_output_shapes
: 
�
rnn_3/rnn/while/NextIteration_2NextIteration3rnn_3/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
rnn_3/rnn/while/NextIteration_3NextIteration%rnn_3/rnn/while/basic_lstm_cell/Add_1*
T0*
_output_shapes

:

�
rnn_3/rnn/while/NextIteration_4NextIteration%rnn_3/rnn/while/basic_lstm_cell/Mul_2*
T0*
_output_shapes

:

U
rnn_3/rnn/while/ExitExitrnn_3/rnn/while/Switch*
T0*
_output_shapes
: 
Y
rnn_3/rnn/while/Exit_1Exitrnn_3/rnn/while/Switch_1*
T0*
_output_shapes
: 
Y
rnn_3/rnn/while/Exit_2Exitrnn_3/rnn/while/Switch_2*
T0*
_output_shapes
: 
a
rnn_3/rnn/while/Exit_3Exitrnn_3/rnn/while/Switch_3*
T0*
_output_shapes

:

a
rnn_3/rnn/while/Exit_4Exitrnn_3/rnn/while/Switch_4*
T0*
_output_shapes

:

�
,rnn_3/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn_3/rnn/TensorArrayrnn_3/rnn/while/Exit_2*(
_class
loc:@rnn_3/rnn/TensorArray*
_output_shapes
: 
�
&rnn_3/rnn/TensorArrayStack/range/startConst*(
_class
loc:@rnn_3/rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 
�
&rnn_3/rnn/TensorArrayStack/range/deltaConst*(
_class
loc:@rnn_3/rnn/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
�
 rnn_3/rnn/TensorArrayStack/rangeRange&rnn_3/rnn/TensorArrayStack/range/start,rnn_3/rnn/TensorArrayStack/TensorArraySizeV3&rnn_3/rnn/TensorArrayStack/range/delta*

Tidx0*(
_class
loc:@rnn_3/rnn/TensorArray*#
_output_shapes
:���������
�
.rnn_3/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn_3/rnn/TensorArray rnn_3/rnn/TensorArrayStack/rangernn_3/rnn/while/Exit_2*(
_class
loc:@rnn_3/rnn/TensorArray*
dtype0*"
_output_shapes
:
*
element_shape
:

[
rnn_3/rnn/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
R
rnn_3/rnn/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
Y
rnn_3/rnn/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
Y
rnn_3/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
rnn_3/rnn/range_1Rangernn_3/rnn/range_1/startrnn_3/rnn/Rank_1rnn_3/rnn/range_1/delta*

Tidx0*
_output_shapes
:
l
rnn_3/rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
Y
rnn_3/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_3/rnn/concat_2ConcatV2rnn_3/rnn/concat_2/values_0rnn_3/rnn/range_1rnn_3/rnn/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
rnn_3/rnn/transpose_1	Transpose.rnn_3/rnn/TensorArrayStack/TensorArrayGatherV3rnn_3/rnn/concat_2*
Tperm0*
T0*"
_output_shapes
:

f
rnn_3/Reshape_2/shapeConst*
valueB"����
   *
dtype0*
_output_shapes
:

rnn_3/Reshape_2Reshapernn_3/rnn/transpose_1rnn_3/Reshape_2/shape*
T0*
Tshape0*
_output_shapes

:

�
rnn_3/MatMul_1MatMulrnn_3/Reshape_2layer_1/weights/Variable_1/read*
transpose_b( *
T0*
_output_shapes

:*
transpose_a( 
k
rnn_3/predsAddrnn_3/MatMul_1layer_1/biases/Variable_1/read*
T0*
_output_shapes

:
V
rnn_3/save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�	
rnn_3/save/SaveV2/tensor_namesConst*�
value�B�"Blayer/biases/VariableBlayer/biases/Variable_1Blayer/weights/VariableBlayer/weights/Variable_1Blayer_1/biases/VariableBlayer_1/biases/Variable_1Blayer_1/weights/VariableBlayer_1/weights/Variable_1Brnn/beta1_powerBrnn/beta2_powerBrnn/layer/biases/Variable/AdamB rnn/layer/biases/Variable/Adam_1B rnn/layer/biases/Variable_1/AdamB"rnn/layer/biases/Variable_1/Adam_1Brnn/layer/weights/Variable/AdamB!rnn/layer/weights/Variable/Adam_1B!rnn/layer/weights/Variable_1/AdamB#rnn/layer/weights/Variable_1/Adam_1B rnn/layer_1/biases/Variable/AdamB"rnn/layer_1/biases/Variable/Adam_1B"rnn/layer_1/biases/Variable_1/AdamB$rnn/layer_1/biases/Variable_1/Adam_1B!rnn/layer_1/weights/Variable/AdamB#rnn/layer_1/weights/Variable/Adam_1B#rnn/layer_1/weights/Variable_1/AdamB%rnn/layer_1/weights/Variable_1/Adam_1Brnn/rnn/basic_lstm_cell/biasBrnn/rnn/basic_lstm_cell/kernelB%rnn/rnn/rnn/basic_lstm_cell/bias/AdamB'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1B'rnn/rnn/rnn/basic_lstm_cell/kernel/AdamB)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1Brnn_2/beta1_powerBrnn_2/beta2_power*
dtype0*
_output_shapes
:"
�
"rnn_3/save/SaveV2/shape_and_slicesConst*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:"
�	
rnn_3/save/SaveV2SaveV2rnn_3/save/Constrnn_3/save/SaveV2/tensor_names"rnn_3/save/SaveV2/shape_and_sliceslayer/biases/Variablelayer/biases/Variable_1layer/weights/Variablelayer/weights/Variable_1layer_1/biases/Variablelayer_1/biases/Variable_1layer_1/weights/Variablelayer_1/weights/Variable_1rnn/beta1_powerrnn/beta2_powerrnn/layer/biases/Variable/Adam rnn/layer/biases/Variable/Adam_1 rnn/layer/biases/Variable_1/Adam"rnn/layer/biases/Variable_1/Adam_1rnn/layer/weights/Variable/Adam!rnn/layer/weights/Variable/Adam_1!rnn/layer/weights/Variable_1/Adam#rnn/layer/weights/Variable_1/Adam_1 rnn/layer_1/biases/Variable/Adam"rnn/layer_1/biases/Variable/Adam_1"rnn/layer_1/biases/Variable_1/Adam$rnn/layer_1/biases/Variable_1/Adam_1!rnn/layer_1/weights/Variable/Adam#rnn/layer_1/weights/Variable/Adam_1#rnn/layer_1/weights/Variable_1/Adam%rnn/layer_1/weights/Variable_1/Adam_1rnn/rnn/basic_lstm_cell/biasrnn/rnn/basic_lstm_cell/kernel%rnn/rnn/rnn/basic_lstm_cell/bias/Adam'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1'rnn/rnn/rnn/basic_lstm_cell/kernel/Adam)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn_2/beta1_powerrnn_2/beta2_power*0
dtypes&
$2"
�
rnn_3/save/control_dependencyIdentityrnn_3/save/Const^rnn_3/save/SaveV2*
T0*#
_class
loc:@rnn_3/save/Const*
_output_shapes
: 
�	
!rnn_3/save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�"Blayer/biases/VariableBlayer/biases/Variable_1Blayer/weights/VariableBlayer/weights/Variable_1Blayer_1/biases/VariableBlayer_1/biases/Variable_1Blayer_1/weights/VariableBlayer_1/weights/Variable_1Brnn/beta1_powerBrnn/beta2_powerBrnn/layer/biases/Variable/AdamB rnn/layer/biases/Variable/Adam_1B rnn/layer/biases/Variable_1/AdamB"rnn/layer/biases/Variable_1/Adam_1Brnn/layer/weights/Variable/AdamB!rnn/layer/weights/Variable/Adam_1B!rnn/layer/weights/Variable_1/AdamB#rnn/layer/weights/Variable_1/Adam_1B rnn/layer_1/biases/Variable/AdamB"rnn/layer_1/biases/Variable/Adam_1B"rnn/layer_1/biases/Variable_1/AdamB$rnn/layer_1/biases/Variable_1/Adam_1B!rnn/layer_1/weights/Variable/AdamB#rnn/layer_1/weights/Variable/Adam_1B#rnn/layer_1/weights/Variable_1/AdamB%rnn/layer_1/weights/Variable_1/Adam_1Brnn/rnn/basic_lstm_cell/biasBrnn/rnn/basic_lstm_cell/kernelB%rnn/rnn/rnn/basic_lstm_cell/bias/AdamB'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1B'rnn/rnn/rnn/basic_lstm_cell/kernel/AdamB)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1Brnn_2/beta1_powerBrnn_2/beta2_power*
dtype0*
_output_shapes
:"
�
%rnn_3/save/RestoreV2/shape_and_slicesConst"/device:CPU:0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:"
�
rnn_3/save/RestoreV2	RestoreV2rnn_3/save/Const!rnn_3/save/RestoreV2/tensor_names%rnn_3/save/RestoreV2/shape_and_slices"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::*0
dtypes&
$2"
�
rnn_3/save/AssignAssignlayer/biases/Variablernn_3/save/RestoreV2*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_3/save/Assign_1Assignlayer/biases/Variable_1rnn_3/save/RestoreV2:1*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_3/save/Assign_2Assignlayer/weights/Variablernn_3/save/RestoreV2:2*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_3/save/Assign_3Assignlayer/weights/Variable_1rnn_3/save/RestoreV2:3*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_3/save/Assign_4Assignlayer_1/biases/Variablernn_3/save/RestoreV2:4*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_3/save/Assign_5Assignlayer_1/biases/Variable_1rnn_3/save/RestoreV2:5*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_3/save/Assign_6Assignlayer_1/weights/Variablernn_3/save/RestoreV2:6*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_3/save/Assign_7Assignlayer_1/weights/Variable_1rnn_3/save/RestoreV2:7*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_3/save/Assign_8Assignrnn/beta1_powerrnn_3/save/RestoreV2:8*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn_3/save/Assign_9Assignrnn/beta2_powerrnn_3/save/RestoreV2:9*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn_3/save/Assign_10Assignrnn/layer/biases/Variable/Adamrnn_3/save/RestoreV2:10*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_3/save/Assign_11Assign rnn/layer/biases/Variable/Adam_1rnn_3/save/RestoreV2:11*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn_3/save/Assign_12Assign rnn/layer/biases/Variable_1/Adamrnn_3/save/RestoreV2:12*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_3/save/Assign_13Assign"rnn/layer/biases/Variable_1/Adam_1rnn_3/save/RestoreV2:13*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_3/save/Assign_14Assignrnn/layer/weights/Variable/Adamrnn_3/save/RestoreV2:14*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_3/save/Assign_15Assign!rnn/layer/weights/Variable/Adam_1rnn_3/save/RestoreV2:15*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_3/save/Assign_16Assign!rnn/layer/weights/Variable_1/Adamrnn_3/save/RestoreV2:16*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_3/save/Assign_17Assign#rnn/layer/weights/Variable_1/Adam_1rnn_3/save/RestoreV2:17*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_3/save/Assign_18Assign rnn/layer_1/biases/Variable/Adamrnn_3/save/RestoreV2:18*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_3/save/Assign_19Assign"rnn/layer_1/biases/Variable/Adam_1rnn_3/save/RestoreV2:19*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_3/save/Assign_20Assign"rnn/layer_1/biases/Variable_1/Adamrnn_3/save/RestoreV2:20*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_3/save/Assign_21Assign$rnn/layer_1/biases/Variable_1/Adam_1rnn_3/save/RestoreV2:21*
use_locking(*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_3/save/Assign_22Assign!rnn/layer_1/weights/Variable/Adamrnn_3/save/RestoreV2:22*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_3/save/Assign_23Assign#rnn/layer_1/weights/Variable/Adam_1rnn_3/save/RestoreV2:23*
use_locking(*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_3/save/Assign_24Assign#rnn/layer_1/weights/Variable_1/Adamrnn_3/save/RestoreV2:24*
use_locking(*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_3/save/Assign_25Assign%rnn/layer_1/weights/Variable_1/Adam_1rnn_3/save/RestoreV2:25*
use_locking(*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_3/save/Assign_26Assignrnn/rnn/basic_lstm_cell/biasrnn_3/save/RestoreV2:26*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
rnn_3/save/Assign_27Assignrnn/rnn/basic_lstm_cell/kernelrnn_3/save/RestoreV2:27*
use_locking(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
rnn_3/save/Assign_28Assign%rnn/rnn/rnn/basic_lstm_cell/bias/Adamrnn_3/save/RestoreV2:28*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
rnn_3/save/Assign_29Assign'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1rnn_3/save/RestoreV2:29*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
rnn_3/save/Assign_30Assign'rnn/rnn/rnn/basic_lstm_cell/kernel/Adamrnn_3/save/RestoreV2:30*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
rnn_3/save/Assign_31Assign)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn_3/save/RestoreV2:31*
use_locking(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
rnn_3/save/Assign_32Assignrnn_2/beta1_powerrnn_3/save/RestoreV2:32*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn_3/save/Assign_33Assignrnn_2/beta2_powerrnn_3/save/RestoreV2:33*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn_3/save/restore_allNoOp^rnn_3/save/Assign^rnn_3/save/Assign_1^rnn_3/save/Assign_10^rnn_3/save/Assign_11^rnn_3/save/Assign_12^rnn_3/save/Assign_13^rnn_3/save/Assign_14^rnn_3/save/Assign_15^rnn_3/save/Assign_16^rnn_3/save/Assign_17^rnn_3/save/Assign_18^rnn_3/save/Assign_19^rnn_3/save/Assign_2^rnn_3/save/Assign_20^rnn_3/save/Assign_21^rnn_3/save/Assign_22^rnn_3/save/Assign_23^rnn_3/save/Assign_24^rnn_3/save/Assign_25^rnn_3/save/Assign_26^rnn_3/save/Assign_27^rnn_3/save/Assign_28^rnn_3/save/Assign_29^rnn_3/save/Assign_3^rnn_3/save/Assign_30^rnn_3/save/Assign_31^rnn_3/save/Assign_32^rnn_3/save/Assign_33^rnn_3/save/Assign_4^rnn_3/save/Assign_5^rnn_3/save/Assign_6^rnn_3/save/Assign_7^rnn_3/save/Assign_8^rnn_3/save/Assign_9
s
inputs_2Placeholder* 
shape:���������*
dtype0*+
_output_shapes
:���������
t
	outputs_2Placeholder* 
shape:���������*
dtype0*+
_output_shapes
:���������
t
#layer_2/weights/random_normal/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
g
"layer_2/weights/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
$layer_2/weights/random_normal/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
2layer_2/weights/random_normal/RandomStandardNormalRandomStandardNormal#layer_2/weights/random_normal/shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
!layer_2/weights/random_normal/mulMul2layer_2/weights/random_normal/RandomStandardNormal$layer_2/weights/random_normal/stddev*
T0*
_output_shapes

:

�
layer_2/weights/random_normalAdd!layer_2/weights/random_normal/mul"layer_2/weights/random_normal/mean*
T0*
_output_shapes

:

�
layer_2/weights/Variable
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
layer_2/weights/Variable/AssignAssignlayer_2/weights/Variablelayer_2/weights/random_normal*
use_locking(*
T0*+
_class!
loc:@layer_2/weights/Variable*
validate_shape(*
_output_shapes

:

�
layer_2/weights/Variable/readIdentitylayer_2/weights/Variable*
T0*+
_class!
loc:@layer_2/weights/Variable*
_output_shapes

:

v
%layer_2/weights/random_normal_1/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
i
$layer_2/weights/random_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
k
&layer_2/weights/random_normal_1/stddevConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
4layer_2/weights/random_normal_1/RandomStandardNormalRandomStandardNormal%layer_2/weights/random_normal_1/shape*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
�
#layer_2/weights/random_normal_1/mulMul4layer_2/weights/random_normal_1/RandomStandardNormal&layer_2/weights/random_normal_1/stddev*
T0*
_output_shapes

:

�
layer_2/weights/random_normal_1Add#layer_2/weights/random_normal_1/mul$layer_2/weights/random_normal_1/mean*
T0*
_output_shapes

:

�
layer_2/weights/Variable_1
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
!layer_2/weights/Variable_1/AssignAssignlayer_2/weights/Variable_1layer_2/weights/random_normal_1*
use_locking(*
T0*-
_class#
!loc:@layer_2/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
layer_2/weights/Variable_1/readIdentitylayer_2/weights/Variable_1*
T0*-
_class#
!loc:@layer_2/weights/Variable_1*
_output_shapes

:

a
layer_2/biases/ConstConst*
valueB
*���=*
dtype0*
_output_shapes
:

�
layer_2/biases/Variable
VariableV2*
dtype0*
_output_shapes
:
*
	container *
shape:
*
shared_name 
�
layer_2/biases/Variable/AssignAssignlayer_2/biases/Variablelayer_2/biases/Const*
use_locking(*
T0**
_class 
loc:@layer_2/biases/Variable*
validate_shape(*
_output_shapes
:

�
layer_2/biases/Variable/readIdentitylayer_2/biases/Variable*
T0**
_class 
loc:@layer_2/biases/Variable*
_output_shapes
:

c
layer_2/biases/Const_1Const*
valueB*���=*
dtype0*
_output_shapes
:
�
layer_2/biases/Variable_1
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
 layer_2/biases/Variable_1/AssignAssignlayer_2/biases/Variable_1layer_2/biases/Const_1*
use_locking(*
T0*,
_class"
 loc:@layer_2/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
layer_2/biases/Variable_1/readIdentitylayer_2/biases/Variable_1*
T0*,
_class"
 loc:@layer_2/biases/Variable_1*
_output_shapes
:
d
rnn_4/Reshape/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
w
rnn_4/ReshapeReshapeinputs_2rnn_4/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:���������
�
rnn_4/MatMulMatMulrnn_4/Reshapelayer_2/weights/Variable/read*
transpose_b( *
T0*'
_output_shapes
:���������
*
transpose_a( 
n
	rnn_4/addAddrnn_4/MatMullayer_2/biases/Variable/read*
T0*'
_output_shapes
:���������

j
rnn_4/Reshape_1/shapeConst*!
valueB"����   
   *
dtype0*
_output_shapes
:
�
rnn_4/Reshape_1Reshape	rnn_4/addrnn_4/Reshape_1/shape*
T0*
Tshape0*+
_output_shapes
:���������

l
"rnn_4/BasicLSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
n
$rnn_4/BasicLSTMCellZeroState/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
j
(rnn_4/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
#rnn_4/BasicLSTMCellZeroState/concatConcatV2"rnn_4/BasicLSTMCellZeroState/Const$rnn_4/BasicLSTMCellZeroState/Const_1(rnn_4/BasicLSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
m
(rnn_4/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"rnn_4/BasicLSTMCellZeroState/zerosFill#rnn_4/BasicLSTMCellZeroState/concat(rnn_4/BasicLSTMCellZeroState/zeros/Const*
T0*

index_type0*
_output_shapes

:

n
$rnn_4/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
n
$rnn_4/BasicLSTMCellZeroState/Const_3Const*
valueB:
*
dtype0*
_output_shapes
:
n
$rnn_4/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
n
$rnn_4/BasicLSTMCellZeroState/Const_5Const*
valueB:
*
dtype0*
_output_shapes
:
l
*rnn_4/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
%rnn_4/BasicLSTMCellZeroState/concat_1ConcatV2$rnn_4/BasicLSTMCellZeroState/Const_4$rnn_4/BasicLSTMCellZeroState/Const_5*rnn_4/BasicLSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
o
*rnn_4/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
$rnn_4/BasicLSTMCellZeroState/zeros_1Fill%rnn_4/BasicLSTMCellZeroState/concat_1*rnn_4/BasicLSTMCellZeroState/zeros_1/Const*
T0*

index_type0*
_output_shapes

:

n
$rnn_4/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
n
$rnn_4/BasicLSTMCellZeroState/Const_7Const*
valueB:
*
dtype0*
_output_shapes
:
P
rnn_4/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
W
rnn_4/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
W
rnn_4/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
~
rnn_4/rnn/rangeRangernn_4/rnn/range/startrnn_4/rnn/Rankrnn_4/rnn/range/delta*

Tidx0*
_output_shapes
:
j
rnn_4/rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
W
rnn_4/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_4/rnn/concatConcatV2rnn_4/rnn/concat/values_0rnn_4/rnn/rangernn_4/rnn/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
rnn_4/rnn/transpose	Transposernn_4/Reshape_1rnn_4/rnn/concat*
T0*+
_output_shapes
:���������
*
Tperm0
b
rnn_4/rnn/ShapeShapernn_4/rnn/transpose*
T0*
out_type0*
_output_shapes
:
g
rnn_4/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
rnn_4/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
i
rnn_4/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
rnn_4/rnn/strided_sliceStridedSlicernn_4/rnn/Shapernn_4/rnn/strided_slice/stackrnn_4/rnn/strided_slice/stack_1rnn_4/rnn/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
d
rnn_4/rnn/Shape_1Shapernn_4/rnn/transpose*
T0*
out_type0*
_output_shapes
:
i
rnn_4/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
k
!rnn_4/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
k
!rnn_4/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
rnn_4/rnn/strided_slice_1StridedSlicernn_4/rnn/Shape_1rnn_4/rnn/strided_slice_1/stack!rnn_4/rnn/strided_slice_1/stack_1!rnn_4/rnn/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
d
rnn_4/rnn/Shape_2Shapernn_4/rnn/transpose*
T0*
out_type0*
_output_shapes
:
i
rnn_4/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
k
!rnn_4/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
k
!rnn_4/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
rnn_4/rnn/strided_slice_2StridedSlicernn_4/rnn/Shape_2rnn_4/rnn/strided_slice_2/stack!rnn_4/rnn/strided_slice_2/stack_1!rnn_4/rnn/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Z
rnn_4/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_4/rnn/ExpandDims
ExpandDimsrnn_4/rnn/strided_slice_2rnn_4/rnn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
Y
rnn_4/rnn/ConstConst*
valueB:
*
dtype0*
_output_shapes
:
Y
rnn_4/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_4/rnn/concat_1ConcatV2rnn_4/rnn/ExpandDimsrnn_4/rnn/Constrnn_4/rnn/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
Z
rnn_4/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
rnn_4/rnn/zerosFillrnn_4/rnn/concat_1rnn_4/rnn/zeros/Const*
T0*

index_type0*'
_output_shapes
:���������

P
rnn_4/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_4/rnn/TensorArrayTensorArrayV3rnn_4/rnn/strided_slice_1*$
element_shape:���������
*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*5
tensor_array_name rnn_4/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: 
�
rnn_4/rnn/TensorArray_1TensorArrayV3rnn_4/rnn/strided_slice_1*$
element_shape:���������
*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*4
tensor_array_namernn_4/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: 
u
"rnn_4/rnn/TensorArrayUnstack/ShapeShapernn_4/rnn/transpose*
T0*
out_type0*
_output_shapes
:
z
0rnn_4/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2rnn_4/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2rnn_4/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
*rnn_4/rnn/TensorArrayUnstack/strided_sliceStridedSlice"rnn_4/rnn/TensorArrayUnstack/Shape0rnn_4/rnn/TensorArrayUnstack/strided_slice/stack2rnn_4/rnn/TensorArrayUnstack/strided_slice/stack_12rnn_4/rnn/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
j
(rnn_4/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
j
(rnn_4/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
"rnn_4/rnn/TensorArrayUnstack/rangeRange(rnn_4/rnn/TensorArrayUnstack/range/start*rnn_4/rnn/TensorArrayUnstack/strided_slice(rnn_4/rnn/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:���������
�
Drnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn_4/rnn/TensorArray_1"rnn_4/rnn/TensorArrayUnstack/rangernn_4/rnn/transposernn_4/rnn/TensorArray_1:1*
T0*&
_class
loc:@rnn_4/rnn/transpose*
_output_shapes
: 
U
rnn_4/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
m
rnn_4/rnn/MaximumMaximumrnn_4/rnn/Maximum/xrnn_4/rnn/strided_slice_1*
T0*
_output_shapes
: 
k
rnn_4/rnn/MinimumMinimumrnn_4/rnn/strided_slice_1rnn_4/rnn/Maximum*
T0*
_output_shapes
: 
c
!rnn_4/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_4/rnn/while/EnterEnter!rnn_4/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *-

frame_namernn_4/rnn/while/while_context
�
rnn_4/rnn/while/Enter_1Enterrnn_4/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *-

frame_namernn_4/rnn/while/while_context
�
rnn_4/rnn/while/Enter_2Enterrnn_4/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *-

frame_namernn_4/rnn/while/while_context
�
rnn_4/rnn/while/Enter_3Enter"rnn_4/BasicLSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*-

frame_namernn_4/rnn/while/while_context
�
rnn_4/rnn/while/Enter_4Enter$rnn_4/BasicLSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*-

frame_namernn_4/rnn/while/while_context
�
rnn_4/rnn/while/MergeMergernn_4/rnn/while/Enterrnn_4/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 
�
rnn_4/rnn/while/Merge_1Mergernn_4/rnn/while/Enter_1rnn_4/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
rnn_4/rnn/while/Merge_2Mergernn_4/rnn/while/Enter_2rnn_4/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
�
rnn_4/rnn/while/Merge_3Mergernn_4/rnn/while/Enter_3rnn_4/rnn/while/NextIteration_3*
T0*
N* 
_output_shapes
:
: 
�
rnn_4/rnn/while/Merge_4Mergernn_4/rnn/while/Enter_4rnn_4/rnn/while/NextIteration_4*
T0*
N* 
_output_shapes
:
: 
p
rnn_4/rnn/while/LessLessrnn_4/rnn/while/Mergernn_4/rnn/while/Less/Enter*
T0*
_output_shapes
: 
�
rnn_4/rnn/while/Less/EnterEnterrnn_4/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *-

frame_namernn_4/rnn/while/while_context
v
rnn_4/rnn/while/Less_1Lessrnn_4/rnn/while/Merge_1rnn_4/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
�
rnn_4/rnn/while/Less_1/EnterEnterrnn_4/rnn/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *-

frame_namernn_4/rnn/while/while_context
n
rnn_4/rnn/while/LogicalAnd
LogicalAndrnn_4/rnn/while/Lessrnn_4/rnn/while/Less_1*
_output_shapes
: 
X
rnn_4/rnn/while/LoopCondLoopCondrnn_4/rnn/while/LogicalAnd*
_output_shapes
: 
�
rnn_4/rnn/while/SwitchSwitchrnn_4/rnn/while/Mergernn_4/rnn/while/LoopCond*
T0*(
_class
loc:@rnn_4/rnn/while/Merge*
_output_shapes
: : 
�
rnn_4/rnn/while/Switch_1Switchrnn_4/rnn/while/Merge_1rnn_4/rnn/while/LoopCond*
T0**
_class 
loc:@rnn_4/rnn/while/Merge_1*
_output_shapes
: : 
�
rnn_4/rnn/while/Switch_2Switchrnn_4/rnn/while/Merge_2rnn_4/rnn/while/LoopCond*
T0**
_class 
loc:@rnn_4/rnn/while/Merge_2*
_output_shapes
: : 
�
rnn_4/rnn/while/Switch_3Switchrnn_4/rnn/while/Merge_3rnn_4/rnn/while/LoopCond*
T0**
_class 
loc:@rnn_4/rnn/while/Merge_3*(
_output_shapes
:
:

�
rnn_4/rnn/while/Switch_4Switchrnn_4/rnn/while/Merge_4rnn_4/rnn/while/LoopCond*
T0**
_class 
loc:@rnn_4/rnn/while/Merge_4*(
_output_shapes
:
:

_
rnn_4/rnn/while/IdentityIdentityrnn_4/rnn/while/Switch:1*
T0*
_output_shapes
: 
c
rnn_4/rnn/while/Identity_1Identityrnn_4/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
c
rnn_4/rnn/while/Identity_2Identityrnn_4/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
k
rnn_4/rnn/while/Identity_3Identityrnn_4/rnn/while/Switch_3:1*
T0*
_output_shapes

:

k
rnn_4/rnn/while/Identity_4Identityrnn_4/rnn/while/Switch_4:1*
T0*
_output_shapes

:

r
rnn_4/rnn/while/add/yConst^rnn_4/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
l
rnn_4/rnn/while/addAddrnn_4/rnn/while/Identityrnn_4/rnn/while/add/y*
T0*
_output_shapes
: 
�
!rnn_4/rnn/while/TensorArrayReadV3TensorArrayReadV3'rnn_4/rnn/while/TensorArrayReadV3/Enterrnn_4/rnn/while/Identity_1)rnn_4/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:���������

�
'rnn_4/rnn/while/TensorArrayReadV3/EnterEnterrnn_4/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_4/rnn/while/while_context
�
)rnn_4/rnn/while/TensorArrayReadV3/Enter_1EnterDrnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *-

frame_namernn_4/rnn/while/while_context
�
%rnn_4/rnn/while/basic_lstm_cell/ConstConst^rnn_4/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
+rnn_4/rnn/while/basic_lstm_cell/concat/axisConst^rnn_4/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
&rnn_4/rnn/while/basic_lstm_cell/concatConcatV2!rnn_4/rnn/while/TensorArrayReadV3rnn_4/rnn/while/Identity_4+rnn_4/rnn/while/basic_lstm_cell/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
�
&rnn_4/rnn/while/basic_lstm_cell/MatMulMatMul&rnn_4/rnn/while/basic_lstm_cell/concat,rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter*
T0*
_output_shapes

:(*
transpose_a( *
transpose_b( 
�
,rnn_4/rnn/while/basic_lstm_cell/MatMul/EnterEnter#rnn/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*-

frame_namernn_4/rnn/while/while_context
�
'rnn_4/rnn/while/basic_lstm_cell/BiasAddBiasAdd&rnn_4/rnn/while/basic_lstm_cell/MatMul-rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes

:(
�
-rnn_4/rnn/while/basic_lstm_cell/BiasAdd/EnterEnter!rnn/rnn/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:(*-

frame_namernn_4/rnn/while/while_context
�
'rnn_4/rnn/while/basic_lstm_cell/Const_1Const^rnn_4/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
%rnn_4/rnn/while/basic_lstm_cell/splitSplit%rnn_4/rnn/while/basic_lstm_cell/Const'rnn_4/rnn/while/basic_lstm_cell/BiasAdd*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split
�
'rnn_4/rnn/while/basic_lstm_cell/Const_2Const^rnn_4/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
#rnn_4/rnn/while/basic_lstm_cell/AddAdd'rnn_4/rnn/while/basic_lstm_cell/split:2'rnn_4/rnn/while/basic_lstm_cell/Const_2*
T0*
_output_shapes

:

�
'rnn_4/rnn/while/basic_lstm_cell/SigmoidSigmoid#rnn_4/rnn/while/basic_lstm_cell/Add*
T0*
_output_shapes

:

�
#rnn_4/rnn/while/basic_lstm_cell/MulMulrnn_4/rnn/while/Identity_3'rnn_4/rnn/while/basic_lstm_cell/Sigmoid*
T0*
_output_shapes

:

�
)rnn_4/rnn/while/basic_lstm_cell/Sigmoid_1Sigmoid%rnn_4/rnn/while/basic_lstm_cell/split*
T0*
_output_shapes

:

~
$rnn_4/rnn/while/basic_lstm_cell/TanhTanh'rnn_4/rnn/while/basic_lstm_cell/split:1*
T0*
_output_shapes

:

�
%rnn_4/rnn/while/basic_lstm_cell/Mul_1Mul)rnn_4/rnn/while/basic_lstm_cell/Sigmoid_1$rnn_4/rnn/while/basic_lstm_cell/Tanh*
T0*
_output_shapes

:

�
%rnn_4/rnn/while/basic_lstm_cell/Add_1Add#rnn_4/rnn/while/basic_lstm_cell/Mul%rnn_4/rnn/while/basic_lstm_cell/Mul_1*
T0*
_output_shapes

:

~
&rnn_4/rnn/while/basic_lstm_cell/Tanh_1Tanh%rnn_4/rnn/while/basic_lstm_cell/Add_1*
T0*
_output_shapes

:

�
)rnn_4/rnn/while/basic_lstm_cell/Sigmoid_2Sigmoid'rnn_4/rnn/while/basic_lstm_cell/split:3*
T0*
_output_shapes

:

�
%rnn_4/rnn/while/basic_lstm_cell/Mul_2Mul&rnn_4/rnn/while/basic_lstm_cell/Tanh_1)rnn_4/rnn/while/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes

:

�
3rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV39rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn_4/rnn/while/Identity_1%rnn_4/rnn/while/basic_lstm_cell/Mul_2rnn_4/rnn/while/Identity_2*
T0*8
_class.
,*loc:@rnn_4/rnn/while/basic_lstm_cell/Mul_2*
_output_shapes
: 
�
9rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn_4/rnn/TensorArray*
is_constant(*
_output_shapes
:*-

frame_namernn_4/rnn/while/while_context*
T0*8
_class.
,*loc:@rnn_4/rnn/while/basic_lstm_cell/Mul_2*
parallel_iterations 
t
rnn_4/rnn/while/add_1/yConst^rnn_4/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
r
rnn_4/rnn/while/add_1Addrnn_4/rnn/while/Identity_1rnn_4/rnn/while/add_1/y*
T0*
_output_shapes
: 
d
rnn_4/rnn/while/NextIterationNextIterationrnn_4/rnn/while/add*
T0*
_output_shapes
: 
h
rnn_4/rnn/while/NextIteration_1NextIterationrnn_4/rnn/while/add_1*
T0*
_output_shapes
: 
�
rnn_4/rnn/while/NextIteration_2NextIteration3rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
rnn_4/rnn/while/NextIteration_3NextIteration%rnn_4/rnn/while/basic_lstm_cell/Add_1*
T0*
_output_shapes

:

�
rnn_4/rnn/while/NextIteration_4NextIteration%rnn_4/rnn/while/basic_lstm_cell/Mul_2*
T0*
_output_shapes

:

U
rnn_4/rnn/while/ExitExitrnn_4/rnn/while/Switch*
T0*
_output_shapes
: 
Y
rnn_4/rnn/while/Exit_1Exitrnn_4/rnn/while/Switch_1*
T0*
_output_shapes
: 
Y
rnn_4/rnn/while/Exit_2Exitrnn_4/rnn/while/Switch_2*
T0*
_output_shapes
: 
a
rnn_4/rnn/while/Exit_3Exitrnn_4/rnn/while/Switch_3*
T0*
_output_shapes

:

a
rnn_4/rnn/while/Exit_4Exitrnn_4/rnn/while/Switch_4*
T0*
_output_shapes

:

�
,rnn_4/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn_4/rnn/TensorArrayrnn_4/rnn/while/Exit_2*(
_class
loc:@rnn_4/rnn/TensorArray*
_output_shapes
: 
�
&rnn_4/rnn/TensorArrayStack/range/startConst*(
_class
loc:@rnn_4/rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 
�
&rnn_4/rnn/TensorArrayStack/range/deltaConst*(
_class
loc:@rnn_4/rnn/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
�
 rnn_4/rnn/TensorArrayStack/rangeRange&rnn_4/rnn/TensorArrayStack/range/start,rnn_4/rnn/TensorArrayStack/TensorArraySizeV3&rnn_4/rnn/TensorArrayStack/range/delta*

Tidx0*(
_class
loc:@rnn_4/rnn/TensorArray*#
_output_shapes
:���������
�
.rnn_4/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn_4/rnn/TensorArray rnn_4/rnn/TensorArrayStack/rangernn_4/rnn/while/Exit_2*
element_shape
:
*(
_class
loc:@rnn_4/rnn/TensorArray*
dtype0*"
_output_shapes
:

[
rnn_4/rnn/Const_1Const*
valueB:
*
dtype0*
_output_shapes
:
R
rnn_4/rnn/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
Y
rnn_4/rnn/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
Y
rnn_4/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
rnn_4/rnn/range_1Rangernn_4/rnn/range_1/startrnn_4/rnn/Rank_1rnn_4/rnn/range_1/delta*

Tidx0*
_output_shapes
:
l
rnn_4/rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
Y
rnn_4/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_4/rnn/concat_2ConcatV2rnn_4/rnn/concat_2/values_0rnn_4/rnn/range_1rnn_4/rnn/concat_2/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
rnn_4/rnn/transpose_1	Transpose.rnn_4/rnn/TensorArrayStack/TensorArrayGatherV3rnn_4/rnn/concat_2*
T0*"
_output_shapes
:
*
Tperm0
f
rnn_4/Reshape_2/shapeConst*
valueB"����
   *
dtype0*
_output_shapes
:

rnn_4/Reshape_2Reshapernn_4/rnn/transpose_1rnn_4/Reshape_2/shape*
T0*
Tshape0*
_output_shapes

:


�
rnn_4/MatMul_1MatMulrnn_4/Reshape_2layer_2/weights/Variable_1/read*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a( 
k
rnn_4/predsAddrnn_4/MatMul_1layer_2/biases/Variable_1/read*
T0*
_output_shapes

:

h
rnn_4/Reshape_3/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
q
rnn_4/Reshape_3Reshapernn_4/predsrnn_4/Reshape_3/shape*
T0*
Tshape0*
_output_shapes
:

h
rnn_4/Reshape_4/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
x
rnn_4/Reshape_4Reshape	outputs_2rnn_4/Reshape_4/shape*
T0*
Tshape0*#
_output_shapes
:���������
W
	rnn_4/subSubrnn_4/Reshape_3rnn_4/Reshape_4*
T0*
_output_shapes
:

F
rnn_4/SquareSquare	rnn_4/sub*
T0*
_output_shapes
:

U
rnn_4/ConstConst*
valueB: *
dtype0*
_output_shapes
:
k

rnn_4/MeanMeanrnn_4/Squarernn_4/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
X
rnn_4/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
rnn_4/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
rnn_4/gradients/FillFillrnn_4/gradients/Shapernn_4/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
Y
rnn_4/gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_4/gradients/f_count_1Enterrnn_4/gradients/f_count*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *-

frame_namernn_4/rnn/while/while_context
�
rnn_4/gradients/MergeMergernn_4/gradients/f_count_1rnn_4/gradients/NextIteration*
T0*
N*
_output_shapes
: : 
t
rnn_4/gradients/SwitchSwitchrnn_4/gradients/Mergernn_4/rnn/while/LoopCond*
T0*
_output_shapes
: : 
r
rnn_4/gradients/Add/yConst^rnn_4/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
l
rnn_4/gradients/AddAddrnn_4/gradients/Switch:1rnn_4/gradients/Add/y*
T0*
_output_shapes
: 
�
rnn_4/gradients/NextIterationNextIterationrnn_4/gradients/Addg^rnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2Q^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2K^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2M^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2K^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2M^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2I^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2K^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2O^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2Q^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1*
T0*
_output_shapes
: 
Z
rnn_4/gradients/f_count_2Exitrnn_4/gradients/Switch*
T0*
_output_shapes
: 
Y
rnn_4/gradients/b_countConst*
value	B :*
dtype0*
_output_shapes
: 
�
rnn_4/gradients/b_count_1Enterrnn_4/gradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
rnn_4/gradients/Merge_1Mergernn_4/gradients/b_count_1rnn_4/gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
rnn_4/gradients/GreaterEqualGreaterEqualrnn_4/gradients/Merge_1"rnn_4/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
�
"rnn_4/gradients/GreaterEqual/EnterEnterrnn_4/gradients/b_count*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
[
rnn_4/gradients/b_count_2LoopCondrnn_4/gradients/GreaterEqual*
_output_shapes
: 
y
rnn_4/gradients/Switch_1Switchrnn_4/gradients/Merge_1rnn_4/gradients/b_count_2*
T0*
_output_shapes
: : 
{
rnn_4/gradients/SubSubrnn_4/gradients/Switch_1:1"rnn_4/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
�
rnn_4/gradients/NextIteration_1NextIterationrnn_4/gradients/Subb^rnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
\
rnn_4/gradients/b_count_3Exitrnn_4/gradients/Switch_1*
T0*
_output_shapes
: 
w
-rnn_4/gradients/rnn_4/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
'rnn_4/gradients/rnn_4/Mean_grad/ReshapeReshapernn_4/gradients/Fill-rnn_4/gradients/rnn_4/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
o
%rnn_4/gradients/rnn_4/Mean_grad/ConstConst*
valueB:
*
dtype0*
_output_shapes
:
�
$rnn_4/gradients/rnn_4/Mean_grad/TileTile'rnn_4/gradients/rnn_4/Mean_grad/Reshape%rnn_4/gradients/rnn_4/Mean_grad/Const*
T0*
_output_shapes
:
*

Tmultiples0
l
'rnn_4/gradients/rnn_4/Mean_grad/Const_1Const*
valueB
 *   A*
dtype0*
_output_shapes
: 
�
'rnn_4/gradients/rnn_4/Mean_grad/truedivRealDiv$rnn_4/gradients/rnn_4/Mean_grad/Tile'rnn_4/gradients/rnn_4/Mean_grad/Const_1*
T0*
_output_shapes
:

�
'rnn_4/gradients/rnn_4/Square_grad/ConstConst(^rnn_4/gradients/rnn_4/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
%rnn_4/gradients/rnn_4/Square_grad/MulMul	rnn_4/sub'rnn_4/gradients/rnn_4/Square_grad/Const*
T0*
_output_shapes
:

�
'rnn_4/gradients/rnn_4/Square_grad/Mul_1Mul'rnn_4/gradients/rnn_4/Mean_grad/truediv%rnn_4/gradients/rnn_4/Square_grad/Mul*
T0*
_output_shapes
:

n
$rnn_4/gradients/rnn_4/sub_grad/ShapeConst*
valueB:
*
dtype0*
_output_shapes
:
u
&rnn_4/gradients/rnn_4/sub_grad/Shape_1Shapernn_4/Reshape_4*
T0*
out_type0*
_output_shapes
:
�
4rnn_4/gradients/rnn_4/sub_grad/BroadcastGradientArgsBroadcastGradientArgs$rnn_4/gradients/rnn_4/sub_grad/Shape&rnn_4/gradients/rnn_4/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
"rnn_4/gradients/rnn_4/sub_grad/SumSum'rnn_4/gradients/rnn_4/Square_grad/Mul_14rnn_4/gradients/rnn_4/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
&rnn_4/gradients/rnn_4/sub_grad/ReshapeReshape"rnn_4/gradients/rnn_4/sub_grad/Sum$rnn_4/gradients/rnn_4/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:

�
$rnn_4/gradients/rnn_4/sub_grad/Sum_1Sum'rnn_4/gradients/rnn_4/Square_grad/Mul_16rnn_4/gradients/rnn_4/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
r
"rnn_4/gradients/rnn_4/sub_grad/NegNeg$rnn_4/gradients/rnn_4/sub_grad/Sum_1*
T0*
_output_shapes
:
�
(rnn_4/gradients/rnn_4/sub_grad/Reshape_1Reshape"rnn_4/gradients/rnn_4/sub_grad/Neg&rnn_4/gradients/rnn_4/sub_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������
�
/rnn_4/gradients/rnn_4/sub_grad/tuple/group_depsNoOp'^rnn_4/gradients/rnn_4/sub_grad/Reshape)^rnn_4/gradients/rnn_4/sub_grad/Reshape_1
�
7rnn_4/gradients/rnn_4/sub_grad/tuple/control_dependencyIdentity&rnn_4/gradients/rnn_4/sub_grad/Reshape0^rnn_4/gradients/rnn_4/sub_grad/tuple/group_deps*
T0*9
_class/
-+loc:@rnn_4/gradients/rnn_4/sub_grad/Reshape*
_output_shapes
:

�
9rnn_4/gradients/rnn_4/sub_grad/tuple/control_dependency_1Identity(rnn_4/gradients/rnn_4/sub_grad/Reshape_10^rnn_4/gradients/rnn_4/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@rnn_4/gradients/rnn_4/sub_grad/Reshape_1*#
_output_shapes
:���������
{
*rnn_4/gradients/rnn_4/Reshape_3_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
�
,rnn_4/gradients/rnn_4/Reshape_3_grad/ReshapeReshape7rnn_4/gradients/rnn_4/sub_grad/tuple/control_dependency*rnn_4/gradients/rnn_4/Reshape_3_grad/Shape*
T0*
Tshape0*
_output_shapes

:

w
&rnn_4/gradients/rnn_4/preds_grad/ShapeConst*
valueB"
      *
dtype0*
_output_shapes
:
r
(rnn_4/gradients/rnn_4/preds_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6rnn_4/gradients/rnn_4/preds_grad/BroadcastGradientArgsBroadcastGradientArgs&rnn_4/gradients/rnn_4/preds_grad/Shape(rnn_4/gradients/rnn_4/preds_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$rnn_4/gradients/rnn_4/preds_grad/SumSum,rnn_4/gradients/rnn_4/Reshape_3_grad/Reshape6rnn_4/gradients/rnn_4/preds_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

�
(rnn_4/gradients/rnn_4/preds_grad/ReshapeReshape$rnn_4/gradients/rnn_4/preds_grad/Sum&rnn_4/gradients/rnn_4/preds_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
&rnn_4/gradients/rnn_4/preds_grad/Sum_1Sum,rnn_4/gradients/rnn_4/Reshape_3_grad/Reshape8rnn_4/gradients/rnn_4/preds_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
*rnn_4/gradients/rnn_4/preds_grad/Reshape_1Reshape&rnn_4/gradients/rnn_4/preds_grad/Sum_1(rnn_4/gradients/rnn_4/preds_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1rnn_4/gradients/rnn_4/preds_grad/tuple/group_depsNoOp)^rnn_4/gradients/rnn_4/preds_grad/Reshape+^rnn_4/gradients/rnn_4/preds_grad/Reshape_1
�
9rnn_4/gradients/rnn_4/preds_grad/tuple/control_dependencyIdentity(rnn_4/gradients/rnn_4/preds_grad/Reshape2^rnn_4/gradients/rnn_4/preds_grad/tuple/group_deps*
T0*;
_class1
/-loc:@rnn_4/gradients/rnn_4/preds_grad/Reshape*
_output_shapes

:

�
;rnn_4/gradients/rnn_4/preds_grad/tuple/control_dependency_1Identity*rnn_4/gradients/rnn_4/preds_grad/Reshape_12^rnn_4/gradients/rnn_4/preds_grad/tuple/group_deps*
T0*=
_class3
1/loc:@rnn_4/gradients/rnn_4/preds_grad/Reshape_1*
_output_shapes
:
�
*rnn_4/gradients/rnn_4/MatMul_1_grad/MatMulMatMul9rnn_4/gradients/rnn_4/preds_grad/tuple/control_dependencylayer_2/weights/Variable_1/read*
transpose_b(*
T0*
_output_shapes

:

*
transpose_a( 
�
,rnn_4/gradients/rnn_4/MatMul_1_grad/MatMul_1MatMulrnn_4/Reshape_29rnn_4/gradients/rnn_4/preds_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a(
�
4rnn_4/gradients/rnn_4/MatMul_1_grad/tuple/group_depsNoOp+^rnn_4/gradients/rnn_4/MatMul_1_grad/MatMul-^rnn_4/gradients/rnn_4/MatMul_1_grad/MatMul_1
�
<rnn_4/gradients/rnn_4/MatMul_1_grad/tuple/control_dependencyIdentity*rnn_4/gradients/rnn_4/MatMul_1_grad/MatMul5^rnn_4/gradients/rnn_4/MatMul_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@rnn_4/gradients/rnn_4/MatMul_1_grad/MatMul*
_output_shapes

:


�
>rnn_4/gradients/rnn_4/MatMul_1_grad/tuple/control_dependency_1Identity,rnn_4/gradients/rnn_4/MatMul_1_grad/MatMul_15^rnn_4/gradients/rnn_4/MatMul_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@rnn_4/gradients/rnn_4/MatMul_1_grad/MatMul_1*
_output_shapes

:


*rnn_4/gradients/rnn_4/Reshape_2_grad/ShapeConst*!
valueB"      
   *
dtype0*
_output_shapes
:
�
,rnn_4/gradients/rnn_4/Reshape_2_grad/ReshapeReshape<rnn_4/gradients/rnn_4/MatMul_1_grad/tuple/control_dependency*rnn_4/gradients/rnn_4/Reshape_2_grad/Shape*
T0*
Tshape0*"
_output_shapes
:

�
<rnn_4/gradients/rnn_4/rnn/transpose_1_grad/InvertPermutationInvertPermutationrnn_4/rnn/concat_2*
T0*
_output_shapes
:
�
4rnn_4/gradients/rnn_4/rnn/transpose_1_grad/transpose	Transpose,rnn_4/gradients/rnn_4/Reshape_2_grad/Reshape<rnn_4/gradients/rnn_4/rnn/transpose_1_grad/InvertPermutation*
Tperm0*
T0*"
_output_shapes
:

�
ernn_4/gradients/rnn_4/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn_4/rnn/TensorArrayrnn_4/rnn/while/Exit_2*(
_class
loc:@rnn_4/rnn/TensorArray*
sourcernn_4/gradients*
_output_shapes

:: 
�
arnn_4/gradients/rnn_4/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityrnn_4/rnn/while/Exit_2f^rnn_4/gradients/rnn_4/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*(
_class
loc:@rnn_4/rnn/TensorArray*
_output_shapes
: 
�
krnn_4/gradients/rnn_4/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3ernn_4/gradients/rnn_4/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3 rnn_4/rnn/TensorArrayStack/range4rnn_4/gradients/rnn_4/rnn/transpose_1_grad/transposearnn_4/gradients/rnn_4/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
j
rnn_4/gradients/zerosConst*
valueB
*    *
dtype0*
_output_shapes

:

l
rnn_4/gradients/zeros_1Const*
valueB
*    *
dtype0*
_output_shapes

:

�
2rnn_4/gradients/rnn_4/rnn/while/Exit_2_grad/b_exitEnterkrnn_4/gradients/rnn_4/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
2rnn_4/gradients/rnn_4/rnn/while/Exit_3_grad/b_exitEnterrnn_4/gradients/zeros*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
2rnn_4/gradients/rnn_4/rnn/while/Exit_4_grad/b_exitEnterrnn_4/gradients/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
6rnn_4/gradients/rnn_4/rnn/while/Switch_2_grad/b_switchMerge2rnn_4/gradients/rnn_4/rnn/while/Exit_2_grad/b_exit=rnn_4/gradients/rnn_4/rnn/while/Switch_2_grad_1/NextIteration*
T0*
N*
_output_shapes
: : 
�
6rnn_4/gradients/rnn_4/rnn/while/Switch_3_grad/b_switchMerge2rnn_4/gradients/rnn_4/rnn/while/Exit_3_grad/b_exit=rnn_4/gradients/rnn_4/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N* 
_output_shapes
:
: 
�
6rnn_4/gradients/rnn_4/rnn/while/Switch_4_grad/b_switchMerge2rnn_4/gradients/rnn_4/rnn/while/Exit_4_grad/b_exit=rnn_4/gradients/rnn_4/rnn/while/Switch_4_grad_1/NextIteration*
T0*
N* 
_output_shapes
:
: 
�
3rnn_4/gradients/rnn_4/rnn/while/Merge_2_grad/SwitchSwitch6rnn_4/gradients/rnn_4/rnn/while/Switch_2_grad/b_switchrnn_4/gradients/b_count_2*
T0*I
_class?
=;loc:@rnn_4/gradients/rnn_4/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: : 
{
=rnn_4/gradients/rnn_4/rnn/while/Merge_2_grad/tuple/group_depsNoOp4^rnn_4/gradients/rnn_4/rnn/while/Merge_2_grad/Switch
�
Ernn_4/gradients/rnn_4/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity3rnn_4/gradients/rnn_4/rnn/while/Merge_2_grad/Switch>^rnn_4/gradients/rnn_4/rnn/while/Merge_2_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_4/gradients/rnn_4/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
�
Grnn_4/gradients/rnn_4/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity5rnn_4/gradients/rnn_4/rnn/while/Merge_2_grad/Switch:1>^rnn_4/gradients/rnn_4/rnn/while/Merge_2_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_4/gradients/rnn_4/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
�
3rnn_4/gradients/rnn_4/rnn/while/Merge_3_grad/SwitchSwitch6rnn_4/gradients/rnn_4/rnn/while/Switch_3_grad/b_switchrnn_4/gradients/b_count_2*
T0*I
_class?
=;loc:@rnn_4/gradients/rnn_4/rnn/while/Switch_3_grad/b_switch*(
_output_shapes
:
:

{
=rnn_4/gradients/rnn_4/rnn/while/Merge_3_grad/tuple/group_depsNoOp4^rnn_4/gradients/rnn_4/rnn/while/Merge_3_grad/Switch
�
Ernn_4/gradients/rnn_4/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity3rnn_4/gradients/rnn_4/rnn/while/Merge_3_grad/Switch>^rnn_4/gradients/rnn_4/rnn/while/Merge_3_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_4/gradients/rnn_4/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
Grnn_4/gradients/rnn_4/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity5rnn_4/gradients/rnn_4/rnn/while/Merge_3_grad/Switch:1>^rnn_4/gradients/rnn_4/rnn/while/Merge_3_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_4/gradients/rnn_4/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
3rnn_4/gradients/rnn_4/rnn/while/Merge_4_grad/SwitchSwitch6rnn_4/gradients/rnn_4/rnn/while/Switch_4_grad/b_switchrnn_4/gradients/b_count_2*
T0*I
_class?
=;loc:@rnn_4/gradients/rnn_4/rnn/while/Switch_4_grad/b_switch*(
_output_shapes
:
:

{
=rnn_4/gradients/rnn_4/rnn/while/Merge_4_grad/tuple/group_depsNoOp4^rnn_4/gradients/rnn_4/rnn/while/Merge_4_grad/Switch
�
Ernn_4/gradients/rnn_4/rnn/while/Merge_4_grad/tuple/control_dependencyIdentity3rnn_4/gradients/rnn_4/rnn/while/Merge_4_grad/Switch>^rnn_4/gradients/rnn_4/rnn/while/Merge_4_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_4/gradients/rnn_4/rnn/while/Switch_4_grad/b_switch*
_output_shapes

:

�
Grnn_4/gradients/rnn_4/rnn/while/Merge_4_grad/tuple/control_dependency_1Identity5rnn_4/gradients/rnn_4/rnn/while/Merge_4_grad/Switch:1>^rnn_4/gradients/rnn_4/rnn/while/Merge_4_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_4/gradients/rnn_4/rnn/while/Switch_4_grad/b_switch*
_output_shapes

:

�
1rnn_4/gradients/rnn_4/rnn/while/Enter_2_grad/ExitExitErnn_4/gradients/rnn_4/rnn/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes
: 
�
1rnn_4/gradients/rnn_4/rnn/while/Enter_3_grad/ExitExitErnn_4/gradients/rnn_4/rnn/while/Merge_3_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
1rnn_4/gradients/rnn_4/rnn/while/Enter_4_grad/ExitExitErnn_4/gradients/rnn_4/rnn/while/Merge_4_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
jrnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3prnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterGrnn_4/gradients/rnn_4/rnn/while/Merge_2_grad/tuple/control_dependency_1*8
_class.
,*loc:@rnn_4/rnn/while/basic_lstm_cell/Mul_2*
sourcernn_4/gradients*
_output_shapes

:: 
�
prnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn_4/rnn/TensorArray*
T0*8
_class.
,*loc:@rnn_4/rnn/while/basic_lstm_cell/Mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
frnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityGrnn_4/gradients/rnn_4/rnn/while/Merge_2_grad/tuple/control_dependency_1k^rnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*8
_class.
,*loc:@rnn_4/rnn/while/basic_lstm_cell/Mul_2*
_output_shapes
: 
�
Zrnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3jrnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3ernn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2frnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*'
_output_shapes
:���������

�
`rnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*-
_class#
!loc:@rnn_4/rnn/while/Identity_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
`rnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2`rnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*
	elem_type0*-
_class#
!loc:@rnn_4/rnn/while/Identity_1*

stack_name *
_output_shapes
:
�
`rnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnter`rnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_4/rnn/while/while_context
�
frnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2`rnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterrnn_4/rnn/while/Identity_1^rnn_4/gradients/Add*
T0*
_output_shapes
: *
swap_memory( 
�
ernn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2krnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^rnn_4/gradients/Sub*
_output_shapes
: *
	elem_type0
�
krnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnter`rnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
arnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerf^rnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2P^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2J^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2L^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2J^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2L^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2H^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2J^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2N^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2P^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
�
Yrnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpH^rnn_4/gradients/rnn_4/rnn/while/Merge_2_grad/tuple/control_dependency_1[^rnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
arnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityZrnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3Z^rnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*m
_classc
a_loc:@rnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
_output_shapes

:

�
crnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityGrnn_4/gradients/rnn_4/rnn/while/Merge_2_grad/tuple/control_dependency_1Z^rnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_4/gradients/rnn_4/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
�
rnn_4/gradients/AddNAddNGrnn_4/gradients/rnn_4/rnn/while/Merge_4_grad/tuple/control_dependency_1arnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*I
_class?
=;loc:@rnn_4/gradients/rnn_4/rnn/while/Switch_4_grad/b_switch*
N*
_output_shapes

:

�
>rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/MulMulrnn_4/gradients/AddNIrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2*
T0*
_output_shapes

:

�
Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/ConstConst*<
_class2
0.loc:@rnn_4/rnn/while/basic_lstm_cell/Sigmoid_2*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_accStackV2Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Const*<
_class2
0.loc:@rnn_4/rnn/while/basic_lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:*
	elem_type0
�
Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/EnterEnterDrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_4/rnn/while/while_context
�
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2StackPushV2Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Enter)rnn_4/rnn/while/basic_lstm_cell/Sigmoid_2^rnn_4/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Irnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2
StackPopV2Ornn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/Enter^rnn_4/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Ornn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/EnterEnterDrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1Mulrnn_4/gradients/AddNKrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2*
T0*
_output_shapes

:

�
Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/ConstConst*9
_class/
-+loc:@rnn_4/rnn/while/basic_lstm_cell/Tanh_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_accStackV2Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Const*
	elem_type0*9
_class/
-+loc:@rnn_4/rnn/while/basic_lstm_cell/Tanh_1*

stack_name *
_output_shapes
:
�
Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/EnterEnterFrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_4/rnn/while/while_context
�
Lrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2StackPushV2Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter&rnn_4/rnn/while/basic_lstm_cell/Tanh_1^rnn_4/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Krnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2
StackPopV2Qrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/Enter^rnn_4/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Qrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/EnterEnterFrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
Krnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_depsNoOp?^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/MulA^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1
�
Srnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependencyIdentity>rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/MulL^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul*
_output_shapes

:

�
Urnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1Identity@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1L^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*S
_classI
GEloc:@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1*
_output_shapes

:

�
Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradKrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2Srnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradIrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2Urnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
=rnn_4/gradients/rnn_4/rnn/while/Switch_2_grad_1/NextIterationNextIterationcrnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
rnn_4/gradients/AddN_1AddNGrnn_4/gradients/rnn_4/rnn/while/Merge_3_grad/tuple/control_dependency_1Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*I
_class?
=;loc:@rnn_4/gradients/rnn_4/rnn/while/Switch_3_grad/b_switch*
N*
_output_shapes

:

l
Krnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_depsNoOp^rnn_4/gradients/AddN_1
�
Srnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyIdentityrnn_4/gradients/AddN_1L^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_4/gradients/rnn_4/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
Urnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Identityrnn_4/gradients/AddN_1L^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@rnn_4/gradients/rnn_4/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
<rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/MulMulSrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyGrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2*
T0*
_output_shapes

:

�
Brnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/ConstConst*:
_class0
.,loc:@rnn_4/rnn/while/basic_lstm_cell/Sigmoid*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Brnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_accStackV2Brnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/Const*
	elem_type0*:
_class0
.,loc:@rnn_4/rnn/while/basic_lstm_cell/Sigmoid*

stack_name *
_output_shapes
:
�
Brnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/EnterEnterBrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_4/rnn/while/while_context
�
Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2StackPushV2Brnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/Enter'rnn_4/rnn/while/basic_lstm_cell/Sigmoid^rnn_4/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Grnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2
StackPopV2Mrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/Enter^rnn_4/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Mrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/EnterEnterBrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
>rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1MulSrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyIrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2*
T0*
_output_shapes

:

�
Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/ConstConst*-
_class#
!loc:@rnn_4/rnn/while/Identity_3*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_accStackV2Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Const*
	elem_type0*-
_class#
!loc:@rnn_4/rnn/while/Identity_3*

stack_name *
_output_shapes
:
�
Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/EnterEnterDrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_4/rnn/while/while_context
�
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2StackPushV2Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Enterrnn_4/rnn/while/Identity_3^rnn_4/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Irnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
StackPopV2Ornn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/Enter^rnn_4/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Ornn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/EnterEnterDrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
Irnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_depsNoOp=^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul?^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1
�
Qrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependencyIdentity<rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/MulJ^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul*
_output_shapes

:

�
Srnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1Identity>rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1J^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1*
_output_shapes

:

�
>rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/MulMulUrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Irnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2*
T0*
_output_shapes

:

�
Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/ConstConst*7
_class-
+)loc:@rnn_4/rnn/while/basic_lstm_cell/Tanh*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_accStackV2Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Const*
	elem_type0*7
_class-
+)loc:@rnn_4/rnn/while/basic_lstm_cell/Tanh*

stack_name *
_output_shapes
:
�
Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/EnterEnterDrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_4/rnn/while/while_context
�
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2StackPushV2Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Enter$rnn_4/rnn/while/basic_lstm_cell/Tanh^rnn_4/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Irnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2
StackPopV2Ornn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/Enter^rnn_4/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Ornn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/EnterEnterDrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1MulUrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Krnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2*
T0*
_output_shapes

:

�
Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/ConstConst*<
_class2
0.loc:@rnn_4/rnn/while/basic_lstm_cell/Sigmoid_1*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_accStackV2Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Const*
	elem_type0*<
_class2
0.loc:@rnn_4/rnn/while/basic_lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:
�
Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/EnterEnterFrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_4/rnn/while/while_context
�
Lrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2StackPushV2Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter)rnn_4/rnn/while/basic_lstm_cell/Sigmoid_1^rnn_4/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Krnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2
StackPopV2Qrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/Enter^rnn_4/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Qrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/EnterEnterFrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
Krnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_depsNoOp?^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/MulA^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1
�
Srnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependencyIdentity>rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/MulL^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul*
_output_shapes

:

�
Urnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1Identity@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1L^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*S
_classI
GEloc:@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1*
_output_shapes

:

�
Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradGrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2Srnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradKrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2Srnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
Brnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Tanh_grad/TanhGradTanhGradIrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2Urnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
=rnn_4/gradients/rnn_4/rnn/while/Switch_3_grad_1/NextIterationNextIterationQrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
>rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/ShapeConst^rnn_4/gradients/Sub*
valueB"   
   *
dtype0*
_output_shapes
:
�
@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/Shape_1Const^rnn_4/gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
�
Nrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgs>rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/Shape@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/SumSumHrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradNrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes

:

�
@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/ReshapeReshape<rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/Sum>rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
>rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/Sum_1SumHrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradPrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
Brnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/Reshape_1Reshape>rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/Sum_1@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Irnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/tuple/group_depsNoOpA^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/ReshapeC^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/Reshape_1
�
Qrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependencyIdentity@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/ReshapeJ^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*S
_classI
GEloc:@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/Reshape*
_output_shapes

:

�
Srnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependency_1IdentityBrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/Reshape_1J^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*U
_classK
IGloc:@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/Reshape_1*
_output_shapes
: 
�
Arnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/split_grad/concatConcatV2Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradBrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Tanh_grad/TanhGradQrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependencyJrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradGrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/split_grad/concat/Const*

Tidx0*
T0*
N*
_output_shapes

:(
�
Grnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/split_grad/concat/ConstConst^rnn_4/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradArnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:(
�
Mrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOpI^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradB^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/split_grad/concat
�
Urnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityArnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/split_grad/concatN^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/split_grad/concat*
_output_shapes

:(
�
Wrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityHrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradN^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes
:(
�
Brnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMulMatMulUrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyHrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul/Enter*
T0*
_output_shapes

:*
transpose_a( *
transpose_b(
�
Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter#rnn/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes

:(*=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1MatMulOrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Urnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:(*
transpose_a(
�
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/ConstConst*9
_class/
-+loc:@rnn_4/rnn/while/basic_lstm_cell/concat*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Const*
	elem_type0*9
_class/
-+loc:@rnn_4/rnn/while/basic_lstm_cell/concat*

stack_name *
_output_shapes
:
�
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnterJrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_4/rnn/while/while_context
�
Prnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter&rnn_4/rnn/while/basic_lstm_cell/concat^rnn_4/gradients/Add*
T0*
_output_shapes

:*
swap_memory( 
�
Ornn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Urnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^rnn_4/gradients/Sub*
_output_shapes

:*
	elem_type0
�
Urnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterJrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
Lrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOpC^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMulE^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1
�
Trnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentityBrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMulM^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul*
_output_shapes

:
�
Vrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityDrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1M^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1*
_output_shapes

:(
�
Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB(*    *
dtype0*
_output_shapes
:(
�
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterHrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:(*=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeJrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Prnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes

:(: 
�
Irnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchJrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2rnn_4/gradients/b_count_2*
T0* 
_output_shapes
:(:(
�
Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/AddAddKrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1Wrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:(
�
Prnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationFrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes
:(
�
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitIrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes
:(
�
Arnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ConstConst^rnn_4/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/RankConst^rnn_4/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
�
?rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/modFloorModArnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/Const@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
�
Arnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeShape!rnn_4/rnn/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
�
Brnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeNShapeNMrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2Ornn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1*
T0*
out_type0*
N* 
_output_shapes
::
�
Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/ConstConst*4
_class*
(&loc:@rnn_4/rnn/while/TensorArrayReadV3*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_accStackV2Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const*
	elem_type0*4
_class*
(&loc:@rnn_4/rnn/while/TensorArrayReadV3*

stack_name *
_output_shapes
:
�
Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/EnterEnterHrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_4/rnn/while/while_context
�
Nrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter!rnn_4/rnn/while/TensorArrayReadV3^rnn_4/gradients/Add*
T0*'
_output_shapes
:���������
*
swap_memory( 
�
Mrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2Srnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^rnn_4/gradients/Sub*
	elem_type0*'
_output_shapes
:���������

�
Srnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterHrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const_1Const*-
_class#
!loc:@rnn_4/rnn/while/Identity_4*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1StackV2Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const_1*
	elem_type0*-
_class#
!loc:@rnn_4/rnn/while/Identity_4*

stack_name *
_output_shapes
:
�
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1EnterJrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*-

frame_namernn_4/rnn/while/while_context
�
Prnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1StackPushV2Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1rnn_4/rnn/while/Identity_4^rnn_4/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Ornn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
StackPopV2Urnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/Enter^rnn_4/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Urnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/EnterEnterJrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffset?rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/modBrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeNDrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
�
Arnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/SliceSliceTrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyHrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ConcatOffsetBrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN*
Index0*
T0*'
_output_shapes
:���������

�
Crnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/Slice_1SliceTrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyJrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ConcatOffset:1Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*
_output_shapes

:

�
Lrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/tuple/group_depsNoOpB^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/SliceD^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/Slice_1
�
Trnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependencyIdentityArnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/SliceM^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/Slice*'
_output_shapes
:���������

�
Vrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1IdentityCrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/Slice_1M^rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*V
_classL
JHloc:@rnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes

:

�
Grnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB(*    *
dtype0*
_output_shapes

:(
�
Irnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1EnterGrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:(*=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
Irnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2MergeIrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Ornn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N* 
_output_shapes
:(: 
�
Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitchIrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2rnn_4/gradients/b_count_2*
T0*(
_output_shapes
:(:(
�
Ernn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter_grad/AddAddJrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Switch:1Vrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:(
�
Ornn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationErnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Add*
T0*
_output_shapes

:(
�
Irnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3ExitHrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Switch*
T0*
_output_shapes

:(
�
Xrnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3^rnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter`rnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^rnn_4/gradients/Sub*:
_class0
.,loc:@rnn_4/rnn/while/TensorArrayReadV3/Enter*
sourcernn_4/gradients*
_output_shapes

:: 
�
^rnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn_4/rnn/TensorArray_1*
is_constant(*
_output_shapes
:*=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context*
T0*:
_class0
.,loc:@rnn_4/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations 
�
`rnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterDrnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*:
_class0
.,loc:@rnn_4/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
: *=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
Trnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentity`rnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1Y^rnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*:
_class0
.,loc:@rnn_4/rnn/while/TensorArrayReadV3/Enter*
_output_shapes
: 
�
Zrnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Xrnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3ernn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Trnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependencyTrnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
�
Drnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Frnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterDrnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *=

frame_name/-rnn_4/gradients/rnn_4/rnn/while/while_context
�
Frnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeFrnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Lrnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
�
Ernn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchFrnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2rnn_4/gradients/b_count_2*
T0*
_output_shapes
: : 
�
Brnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddGrnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Zrnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
�
Lrnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationBrnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
�
Frnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitErnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 
�
=rnn_4/gradients/rnn_4/rnn/while/Switch_4_grad_1/NextIterationNextIterationVrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
{rnn_4/gradients/rnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn_4/rnn/TensorArray_1Frnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3**
_class 
loc:@rnn_4/rnn/TensorArray_1*
sourcernn_4/gradients*
_output_shapes

:: 
�
wrnn_4/gradients/rnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityFrnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3|^rnn_4/gradients/rnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0**
_class 
loc:@rnn_4/rnn/TensorArray_1*
_output_shapes
: 
�
mrnn_4/gradients/rnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3{rnn_4/gradients/rnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3"rnn_4/rnn/TensorArrayUnstack/rangewrnn_4/gradients/rnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*4
_output_shapes"
 :������������������
*
element_shape:
�
jrnn_4/gradients/rnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpn^rnn_4/gradients/rnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3G^rnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
�
rrnn_4/gradients/rnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentitymrnn_4/gradients/rnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3k^rnn_4/gradients/rnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*�
_classv
trloc:@rnn_4/gradients/rnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3*+
_output_shapes
:���������

�
trnn_4/gradients/rnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityFrnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3k^rnn_4/gradients/rnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@rnn_4/gradients/rnn_4/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes
: 
�
:rnn_4/gradients/rnn_4/rnn/transpose_grad/InvertPermutationInvertPermutationrnn_4/rnn/concat*
T0*
_output_shapes
:
�
2rnn_4/gradients/rnn_4/rnn/transpose_grad/transpose	Transposerrnn_4/gradients/rnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency:rnn_4/gradients/rnn_4/rnn/transpose_grad/InvertPermutation*
Tperm0*
T0*+
_output_shapes
:���������

s
*rnn_4/gradients/rnn_4/Reshape_1_grad/ShapeShape	rnn_4/add*
T0*
out_type0*
_output_shapes
:
�
,rnn_4/gradients/rnn_4/Reshape_1_grad/ReshapeReshape2rnn_4/gradients/rnn_4/rnn/transpose_grad/transpose*rnn_4/gradients/rnn_4/Reshape_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

p
$rnn_4/gradients/rnn_4/add_grad/ShapeShapernn_4/MatMul*
T0*
out_type0*
_output_shapes
:
p
&rnn_4/gradients/rnn_4/add_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
�
4rnn_4/gradients/rnn_4/add_grad/BroadcastGradientArgsBroadcastGradientArgs$rnn_4/gradients/rnn_4/add_grad/Shape&rnn_4/gradients/rnn_4/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
"rnn_4/gradients/rnn_4/add_grad/SumSum,rnn_4/gradients/rnn_4/Reshape_1_grad/Reshape4rnn_4/gradients/rnn_4/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
&rnn_4/gradients/rnn_4/add_grad/ReshapeReshape"rnn_4/gradients/rnn_4/add_grad/Sum$rnn_4/gradients/rnn_4/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
$rnn_4/gradients/rnn_4/add_grad/Sum_1Sum,rnn_4/gradients/rnn_4/Reshape_1_grad/Reshape6rnn_4/gradients/rnn_4/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(rnn_4/gradients/rnn_4/add_grad/Reshape_1Reshape$rnn_4/gradients/rnn_4/add_grad/Sum_1&rnn_4/gradients/rnn_4/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

�
/rnn_4/gradients/rnn_4/add_grad/tuple/group_depsNoOp'^rnn_4/gradients/rnn_4/add_grad/Reshape)^rnn_4/gradients/rnn_4/add_grad/Reshape_1
�
7rnn_4/gradients/rnn_4/add_grad/tuple/control_dependencyIdentity&rnn_4/gradients/rnn_4/add_grad/Reshape0^rnn_4/gradients/rnn_4/add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@rnn_4/gradients/rnn_4/add_grad/Reshape*'
_output_shapes
:���������

�
9rnn_4/gradients/rnn_4/add_grad/tuple/control_dependency_1Identity(rnn_4/gradients/rnn_4/add_grad/Reshape_10^rnn_4/gradients/rnn_4/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@rnn_4/gradients/rnn_4/add_grad/Reshape_1*
_output_shapes
:

�
(rnn_4/gradients/rnn_4/MatMul_grad/MatMulMatMul7rnn_4/gradients/rnn_4/add_grad/tuple/control_dependencylayer_2/weights/Variable/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
*rnn_4/gradients/rnn_4/MatMul_grad/MatMul_1MatMulrnn_4/Reshape7rnn_4/gradients/rnn_4/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:
*
transpose_a(
�
2rnn_4/gradients/rnn_4/MatMul_grad/tuple/group_depsNoOp)^rnn_4/gradients/rnn_4/MatMul_grad/MatMul+^rnn_4/gradients/rnn_4/MatMul_grad/MatMul_1
�
:rnn_4/gradients/rnn_4/MatMul_grad/tuple/control_dependencyIdentity(rnn_4/gradients/rnn_4/MatMul_grad/MatMul3^rnn_4/gradients/rnn_4/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@rnn_4/gradients/rnn_4/MatMul_grad/MatMul*'
_output_shapes
:���������
�
<rnn_4/gradients/rnn_4/MatMul_grad/tuple/control_dependency_1Identity*rnn_4/gradients/rnn_4/MatMul_grad/MatMul_13^rnn_4/gradients/rnn_4/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@rnn_4/gradients/rnn_4/MatMul_grad/MatMul_1*
_output_shapes

:

�
rnn_4/beta1_power/initial_valueConst**
_class 
loc:@layer_2/biases/Variable*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
rnn_4/beta1_power
VariableV2**
_class 
loc:@layer_2/biases/Variable*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
rnn_4/beta1_power/AssignAssignrnn_4/beta1_powerrnn_4/beta1_power/initial_value*
use_locking(*
T0**
_class 
loc:@layer_2/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn_4/beta1_power/readIdentityrnn_4/beta1_power*
T0**
_class 
loc:@layer_2/biases/Variable*
_output_shapes
: 
�
rnn_4/beta2_power/initial_valueConst**
_class 
loc:@layer_2/biases/Variable*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
rnn_4/beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name **
_class 
loc:@layer_2/biases/Variable*
	container 
�
rnn_4/beta2_power/AssignAssignrnn_4/beta2_powerrnn_4/beta2_power/initial_value*
use_locking(*
T0**
_class 
loc:@layer_2/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn_4/beta2_power/readIdentityrnn_4/beta2_power*
T0**
_class 
loc:@layer_2/biases/Variable*
_output_shapes
: 
�
3rnn/layer_2/weights/Variable/Adam/Initializer/zerosConst*+
_class!
loc:@layer_2/weights/Variable*
valueB
*    *
dtype0*
_output_shapes

:

�
!rnn/layer_2/weights/Variable/Adam
VariableV2*+
_class!
loc:@layer_2/weights/Variable*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name 
�
(rnn/layer_2/weights/Variable/Adam/AssignAssign!rnn/layer_2/weights/Variable/Adam3rnn/layer_2/weights/Variable/Adam/Initializer/zeros*
T0*+
_class!
loc:@layer_2/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
&rnn/layer_2/weights/Variable/Adam/readIdentity!rnn/layer_2/weights/Variable/Adam*
T0*+
_class!
loc:@layer_2/weights/Variable*
_output_shapes

:

�
5rnn/layer_2/weights/Variable/Adam_1/Initializer/zerosConst*+
_class!
loc:@layer_2/weights/Variable*
valueB
*    *
dtype0*
_output_shapes

:

�
#rnn/layer_2/weights/Variable/Adam_1
VariableV2*
shared_name *+
_class!
loc:@layer_2/weights/Variable*
	container *
shape
:
*
dtype0*
_output_shapes

:

�
*rnn/layer_2/weights/Variable/Adam_1/AssignAssign#rnn/layer_2/weights/Variable/Adam_15rnn/layer_2/weights/Variable/Adam_1/Initializer/zeros*
T0*+
_class!
loc:@layer_2/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
(rnn/layer_2/weights/Variable/Adam_1/readIdentity#rnn/layer_2/weights/Variable/Adam_1*
T0*+
_class!
loc:@layer_2/weights/Variable*
_output_shapes

:

�
5rnn/layer_2/weights/Variable_1/Adam/Initializer/zerosConst*-
_class#
!loc:@layer_2/weights/Variable_1*
valueB
*    *
dtype0*
_output_shapes

:

�
#rnn/layer_2/weights/Variable_1/Adam
VariableV2*-
_class#
!loc:@layer_2/weights/Variable_1*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name 
�
*rnn/layer_2/weights/Variable_1/Adam/AssignAssign#rnn/layer_2/weights/Variable_1/Adam5rnn/layer_2/weights/Variable_1/Adam/Initializer/zeros*
T0*-
_class#
!loc:@layer_2/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
(rnn/layer_2/weights/Variable_1/Adam/readIdentity#rnn/layer_2/weights/Variable_1/Adam*
T0*-
_class#
!loc:@layer_2/weights/Variable_1*
_output_shapes

:

�
7rnn/layer_2/weights/Variable_1/Adam_1/Initializer/zerosConst*-
_class#
!loc:@layer_2/weights/Variable_1*
valueB
*    *
dtype0*
_output_shapes

:

�
%rnn/layer_2/weights/Variable_1/Adam_1
VariableV2*-
_class#
!loc:@layer_2/weights/Variable_1*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name 
�
,rnn/layer_2/weights/Variable_1/Adam_1/AssignAssign%rnn/layer_2/weights/Variable_1/Adam_17rnn/layer_2/weights/Variable_1/Adam_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@layer_2/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
*rnn/layer_2/weights/Variable_1/Adam_1/readIdentity%rnn/layer_2/weights/Variable_1/Adam_1*
T0*-
_class#
!loc:@layer_2/weights/Variable_1*
_output_shapes

:

�
2rnn/layer_2/biases/Variable/Adam/Initializer/zerosConst**
_class 
loc:@layer_2/biases/Variable*
valueB
*    *
dtype0*
_output_shapes
:

�
 rnn/layer_2/biases/Variable/Adam
VariableV2*
dtype0*
_output_shapes
:
*
shared_name **
_class 
loc:@layer_2/biases/Variable*
	container *
shape:

�
'rnn/layer_2/biases/Variable/Adam/AssignAssign rnn/layer_2/biases/Variable/Adam2rnn/layer_2/biases/Variable/Adam/Initializer/zeros*
T0**
_class 
loc:@layer_2/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
%rnn/layer_2/biases/Variable/Adam/readIdentity rnn/layer_2/biases/Variable/Adam*
T0**
_class 
loc:@layer_2/biases/Variable*
_output_shapes
:

�
4rnn/layer_2/biases/Variable/Adam_1/Initializer/zerosConst**
_class 
loc:@layer_2/biases/Variable*
valueB
*    *
dtype0*
_output_shapes
:

�
"rnn/layer_2/biases/Variable/Adam_1
VariableV2*
shared_name **
_class 
loc:@layer_2/biases/Variable*
	container *
shape:
*
dtype0*
_output_shapes
:

�
)rnn/layer_2/biases/Variable/Adam_1/AssignAssign"rnn/layer_2/biases/Variable/Adam_14rnn/layer_2/biases/Variable/Adam_1/Initializer/zeros*
T0**
_class 
loc:@layer_2/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
'rnn/layer_2/biases/Variable/Adam_1/readIdentity"rnn/layer_2/biases/Variable/Adam_1*
T0**
_class 
loc:@layer_2/biases/Variable*
_output_shapes
:

�
4rnn/layer_2/biases/Variable_1/Adam/Initializer/zerosConst*,
_class"
 loc:@layer_2/biases/Variable_1*
valueB*    *
dtype0*
_output_shapes
:
�
"rnn/layer_2/biases/Variable_1/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@layer_2/biases/Variable_1*
	container 
�
)rnn/layer_2/biases/Variable_1/Adam/AssignAssign"rnn/layer_2/biases/Variable_1/Adam4rnn/layer_2/biases/Variable_1/Adam/Initializer/zeros*
T0*,
_class"
 loc:@layer_2/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
'rnn/layer_2/biases/Variable_1/Adam/readIdentity"rnn/layer_2/biases/Variable_1/Adam*
T0*,
_class"
 loc:@layer_2/biases/Variable_1*
_output_shapes
:
�
6rnn/layer_2/biases/Variable_1/Adam_1/Initializer/zerosConst*,
_class"
 loc:@layer_2/biases/Variable_1*
valueB*    *
dtype0*
_output_shapes
:
�
$rnn/layer_2/biases/Variable_1/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@layer_2/biases/Variable_1*
	container *
shape:
�
+rnn/layer_2/biases/Variable_1/Adam_1/AssignAssign$rnn/layer_2/biases/Variable_1/Adam_16rnn/layer_2/biases/Variable_1/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@layer_2/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
)rnn/layer_2/biases/Variable_1/Adam_1/readIdentity$rnn/layer_2/biases/Variable_1/Adam_1*
T0*,
_class"
 loc:@layer_2/biases/Variable_1*
_output_shapes
:
]
rnn_4/Adam/learning_rateConst*
valueB
 *RI9*
dtype0*
_output_shapes
: 
U
rnn_4/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
rnn_4/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
rnn_4/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
:rnn_4/Adam/update_rnn/rnn/basic_lstm_cell/kernel/ApplyAdam	ApplyAdamrnn/rnn/basic_lstm_cell/kernel'rnn/rnn/rnn/basic_lstm_cell/kernel/Adam)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn_4/beta1_power/readrnn_4/beta2_power/readrnn_4/Adam/learning_raternn_4/Adam/beta1rnn_4/Adam/beta2rnn_4/Adam/epsilonIrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
use_locking( *
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
use_nesterov( *
_output_shapes

:(
�
8rnn_4/Adam/update_rnn/rnn/basic_lstm_cell/bias/ApplyAdam	ApplyAdamrnn/rnn/basic_lstm_cell/bias%rnn/rnn/rnn/basic_lstm_cell/bias/Adam'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1rnn_4/beta1_power/readrnn_4/beta2_power/readrnn_4/Adam/learning_raternn_4/Adam/beta1rnn_4/Adam/beta2rnn_4/Adam/epsilonJrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
use_nesterov( *
_output_shapes
:(*
use_locking( 
�
4rnn_4/Adam/update_layer_2/weights/Variable/ApplyAdam	ApplyAdamlayer_2/weights/Variable!rnn/layer_2/weights/Variable/Adam#rnn/layer_2/weights/Variable/Adam_1rnn_4/beta1_power/readrnn_4/beta2_power/readrnn_4/Adam/learning_raternn_4/Adam/beta1rnn_4/Adam/beta2rnn_4/Adam/epsilon<rnn_4/gradients/rnn_4/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@layer_2/weights/Variable*
use_nesterov( *
_output_shapes

:

�
6rnn_4/Adam/update_layer_2/weights/Variable_1/ApplyAdam	ApplyAdamlayer_2/weights/Variable_1#rnn/layer_2/weights/Variable_1/Adam%rnn/layer_2/weights/Variable_1/Adam_1rnn_4/beta1_power/readrnn_4/beta2_power/readrnn_4/Adam/learning_raternn_4/Adam/beta1rnn_4/Adam/beta2rnn_4/Adam/epsilon>rnn_4/gradients/rnn_4/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*-
_class#
!loc:@layer_2/weights/Variable_1*
use_nesterov( *
_output_shapes

:

�
3rnn_4/Adam/update_layer_2/biases/Variable/ApplyAdam	ApplyAdamlayer_2/biases/Variable rnn/layer_2/biases/Variable/Adam"rnn/layer_2/biases/Variable/Adam_1rnn_4/beta1_power/readrnn_4/beta2_power/readrnn_4/Adam/learning_raternn_4/Adam/beta1rnn_4/Adam/beta2rnn_4/Adam/epsilon9rnn_4/gradients/rnn_4/add_grad/tuple/control_dependency_1*
T0**
_class 
loc:@layer_2/biases/Variable*
use_nesterov( *
_output_shapes
:
*
use_locking( 
�
5rnn_4/Adam/update_layer_2/biases/Variable_1/ApplyAdam	ApplyAdamlayer_2/biases/Variable_1"rnn/layer_2/biases/Variable_1/Adam$rnn/layer_2/biases/Variable_1/Adam_1rnn_4/beta1_power/readrnn_4/beta2_power/readrnn_4/Adam/learning_raternn_4/Adam/beta1rnn_4/Adam/beta2rnn_4/Adam/epsilon;rnn_4/gradients/rnn_4/preds_grad/tuple/control_dependency_1*
T0*,
_class"
 loc:@layer_2/biases/Variable_1*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
rnn_4/Adam/mulMulrnn_4/beta1_power/readrnn_4/Adam/beta14^rnn_4/Adam/update_layer_2/biases/Variable/ApplyAdam6^rnn_4/Adam/update_layer_2/biases/Variable_1/ApplyAdam5^rnn_4/Adam/update_layer_2/weights/Variable/ApplyAdam7^rnn_4/Adam/update_layer_2/weights/Variable_1/ApplyAdam9^rnn_4/Adam/update_rnn/rnn/basic_lstm_cell/bias/ApplyAdam;^rnn_4/Adam/update_rnn/rnn/basic_lstm_cell/kernel/ApplyAdam*
T0**
_class 
loc:@layer_2/biases/Variable*
_output_shapes
: 
�
rnn_4/Adam/AssignAssignrnn_4/beta1_powerrnn_4/Adam/mul*
T0**
_class 
loc:@layer_2/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking( 
�
rnn_4/Adam/mul_1Mulrnn_4/beta2_power/readrnn_4/Adam/beta24^rnn_4/Adam/update_layer_2/biases/Variable/ApplyAdam6^rnn_4/Adam/update_layer_2/biases/Variable_1/ApplyAdam5^rnn_4/Adam/update_layer_2/weights/Variable/ApplyAdam7^rnn_4/Adam/update_layer_2/weights/Variable_1/ApplyAdam9^rnn_4/Adam/update_rnn/rnn/basic_lstm_cell/bias/ApplyAdam;^rnn_4/Adam/update_rnn/rnn/basic_lstm_cell/kernel/ApplyAdam*
T0**
_class 
loc:@layer_2/biases/Variable*
_output_shapes
: 
�
rnn_4/Adam/Assign_1Assignrnn_4/beta2_powerrnn_4/Adam/mul_1*
use_locking( *
T0**
_class 
loc:@layer_2/biases/Variable*
validate_shape(*
_output_shapes
: 
�

rnn_4/AdamNoOp^rnn_4/Adam/Assign^rnn_4/Adam/Assign_14^rnn_4/Adam/update_layer_2/biases/Variable/ApplyAdam6^rnn_4/Adam/update_layer_2/biases/Variable_1/ApplyAdam5^rnn_4/Adam/update_layer_2/weights/Variable/ApplyAdam7^rnn_4/Adam/update_layer_2/weights/Variable_1/ApplyAdam9^rnn_4/Adam/update_rnn/rnn/basic_lstm_cell/bias/ApplyAdam;^rnn_4/Adam/update_rnn/rnn/basic_lstm_cell/kernel/ApplyAdam
V
rnn_4/save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
rnn_4/save/SaveV2/tensor_namesConst*�
value�B�0Blayer/biases/VariableBlayer/biases/Variable_1Blayer/weights/VariableBlayer/weights/Variable_1Blayer_1/biases/VariableBlayer_1/biases/Variable_1Blayer_1/weights/VariableBlayer_1/weights/Variable_1Blayer_2/biases/VariableBlayer_2/biases/Variable_1Blayer_2/weights/VariableBlayer_2/weights/Variable_1Brnn/beta1_powerBrnn/beta2_powerBrnn/layer/biases/Variable/AdamB rnn/layer/biases/Variable/Adam_1B rnn/layer/biases/Variable_1/AdamB"rnn/layer/biases/Variable_1/Adam_1Brnn/layer/weights/Variable/AdamB!rnn/layer/weights/Variable/Adam_1B!rnn/layer/weights/Variable_1/AdamB#rnn/layer/weights/Variable_1/Adam_1B rnn/layer_1/biases/Variable/AdamB"rnn/layer_1/biases/Variable/Adam_1B"rnn/layer_1/biases/Variable_1/AdamB$rnn/layer_1/biases/Variable_1/Adam_1B!rnn/layer_1/weights/Variable/AdamB#rnn/layer_1/weights/Variable/Adam_1B#rnn/layer_1/weights/Variable_1/AdamB%rnn/layer_1/weights/Variable_1/Adam_1B rnn/layer_2/biases/Variable/AdamB"rnn/layer_2/biases/Variable/Adam_1B"rnn/layer_2/biases/Variable_1/AdamB$rnn/layer_2/biases/Variable_1/Adam_1B!rnn/layer_2/weights/Variable/AdamB#rnn/layer_2/weights/Variable/Adam_1B#rnn/layer_2/weights/Variable_1/AdamB%rnn/layer_2/weights/Variable_1/Adam_1Brnn/rnn/basic_lstm_cell/biasBrnn/rnn/basic_lstm_cell/kernelB%rnn/rnn/rnn/basic_lstm_cell/bias/AdamB'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1B'rnn/rnn/rnn/basic_lstm_cell/kernel/AdamB)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1Brnn_2/beta1_powerBrnn_2/beta2_powerBrnn_4/beta1_powerBrnn_4/beta2_power*
dtype0*
_output_shapes
:0
�
"rnn_4/save/SaveV2/shape_and_slicesConst*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:0
�
rnn_4/save/SaveV2SaveV2rnn_4/save/Constrnn_4/save/SaveV2/tensor_names"rnn_4/save/SaveV2/shape_and_sliceslayer/biases/Variablelayer/biases/Variable_1layer/weights/Variablelayer/weights/Variable_1layer_1/biases/Variablelayer_1/biases/Variable_1layer_1/weights/Variablelayer_1/weights/Variable_1layer_2/biases/Variablelayer_2/biases/Variable_1layer_2/weights/Variablelayer_2/weights/Variable_1rnn/beta1_powerrnn/beta2_powerrnn/layer/biases/Variable/Adam rnn/layer/biases/Variable/Adam_1 rnn/layer/biases/Variable_1/Adam"rnn/layer/biases/Variable_1/Adam_1rnn/layer/weights/Variable/Adam!rnn/layer/weights/Variable/Adam_1!rnn/layer/weights/Variable_1/Adam#rnn/layer/weights/Variable_1/Adam_1 rnn/layer_1/biases/Variable/Adam"rnn/layer_1/biases/Variable/Adam_1"rnn/layer_1/biases/Variable_1/Adam$rnn/layer_1/biases/Variable_1/Adam_1!rnn/layer_1/weights/Variable/Adam#rnn/layer_1/weights/Variable/Adam_1#rnn/layer_1/weights/Variable_1/Adam%rnn/layer_1/weights/Variable_1/Adam_1 rnn/layer_2/biases/Variable/Adam"rnn/layer_2/biases/Variable/Adam_1"rnn/layer_2/biases/Variable_1/Adam$rnn/layer_2/biases/Variable_1/Adam_1!rnn/layer_2/weights/Variable/Adam#rnn/layer_2/weights/Variable/Adam_1#rnn/layer_2/weights/Variable_1/Adam%rnn/layer_2/weights/Variable_1/Adam_1rnn/rnn/basic_lstm_cell/biasrnn/rnn/basic_lstm_cell/kernel%rnn/rnn/rnn/basic_lstm_cell/bias/Adam'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1'rnn/rnn/rnn/basic_lstm_cell/kernel/Adam)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn_2/beta1_powerrnn_2/beta2_powerrnn_4/beta1_powerrnn_4/beta2_power*>
dtypes4
220
�
rnn_4/save/control_dependencyIdentityrnn_4/save/Const^rnn_4/save/SaveV2*
T0*#
_class
loc:@rnn_4/save/Const*
_output_shapes
: 
�
!rnn_4/save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�0Blayer/biases/VariableBlayer/biases/Variable_1Blayer/weights/VariableBlayer/weights/Variable_1Blayer_1/biases/VariableBlayer_1/biases/Variable_1Blayer_1/weights/VariableBlayer_1/weights/Variable_1Blayer_2/biases/VariableBlayer_2/biases/Variable_1Blayer_2/weights/VariableBlayer_2/weights/Variable_1Brnn/beta1_powerBrnn/beta2_powerBrnn/layer/biases/Variable/AdamB rnn/layer/biases/Variable/Adam_1B rnn/layer/biases/Variable_1/AdamB"rnn/layer/biases/Variable_1/Adam_1Brnn/layer/weights/Variable/AdamB!rnn/layer/weights/Variable/Adam_1B!rnn/layer/weights/Variable_1/AdamB#rnn/layer/weights/Variable_1/Adam_1B rnn/layer_1/biases/Variable/AdamB"rnn/layer_1/biases/Variable/Adam_1B"rnn/layer_1/biases/Variable_1/AdamB$rnn/layer_1/biases/Variable_1/Adam_1B!rnn/layer_1/weights/Variable/AdamB#rnn/layer_1/weights/Variable/Adam_1B#rnn/layer_1/weights/Variable_1/AdamB%rnn/layer_1/weights/Variable_1/Adam_1B rnn/layer_2/biases/Variable/AdamB"rnn/layer_2/biases/Variable/Adam_1B"rnn/layer_2/biases/Variable_1/AdamB$rnn/layer_2/biases/Variable_1/Adam_1B!rnn/layer_2/weights/Variable/AdamB#rnn/layer_2/weights/Variable/Adam_1B#rnn/layer_2/weights/Variable_1/AdamB%rnn/layer_2/weights/Variable_1/Adam_1Brnn/rnn/basic_lstm_cell/biasBrnn/rnn/basic_lstm_cell/kernelB%rnn/rnn/rnn/basic_lstm_cell/bias/AdamB'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1B'rnn/rnn/rnn/basic_lstm_cell/kernel/AdamB)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1Brnn_2/beta1_powerBrnn_2/beta2_powerBrnn_4/beta1_powerBrnn_4/beta2_power*
dtype0*
_output_shapes
:0
�
%rnn_4/save/RestoreV2/shape_and_slicesConst"/device:CPU:0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:0
�
rnn_4/save/RestoreV2	RestoreV2rnn_4/save/Const!rnn_4/save/RestoreV2/tensor_names%rnn_4/save/RestoreV2/shape_and_slices"/device:CPU:0*>
dtypes4
220*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::
�
rnn_4/save/AssignAssignlayer/biases/Variablernn_4/save/RestoreV2*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn_4/save/Assign_1Assignlayer/biases/Variable_1rnn_4/save/RestoreV2:1*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_4/save/Assign_2Assignlayer/weights/Variablernn_4/save/RestoreV2:2*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_4/save/Assign_3Assignlayer/weights/Variable_1rnn_4/save/RestoreV2:3*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_4/save/Assign_4Assignlayer_1/biases/Variablernn_4/save/RestoreV2:4*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_4/save/Assign_5Assignlayer_1/biases/Variable_1rnn_4/save/RestoreV2:5*
use_locking(*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_4/save/Assign_6Assignlayer_1/weights/Variablernn_4/save/RestoreV2:6*
use_locking(*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_4/save/Assign_7Assignlayer_1/weights/Variable_1rnn_4/save/RestoreV2:7*
use_locking(*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_4/save/Assign_8Assignlayer_2/biases/Variablernn_4/save/RestoreV2:8*
T0**
_class 
loc:@layer_2/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn_4/save/Assign_9Assignlayer_2/biases/Variable_1rnn_4/save/RestoreV2:9*
use_locking(*
T0*,
_class"
 loc:@layer_2/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_4/save/Assign_10Assignlayer_2/weights/Variablernn_4/save/RestoreV2:10*
T0*+
_class!
loc:@layer_2/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save/Assign_11Assignlayer_2/weights/Variable_1rnn_4/save/RestoreV2:11*
T0*-
_class#
!loc:@layer_2/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save/Assign_12Assignrnn/beta1_powerrnn_4/save/RestoreV2:12*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(
�
rnn_4/save/Assign_13Assignrnn/beta2_powerrnn_4/save/RestoreV2:13*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(
�
rnn_4/save/Assign_14Assignrnn/layer/biases/Variable/Adamrnn_4/save/RestoreV2:14*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_4/save/Assign_15Assign rnn/layer/biases/Variable/Adam_1rnn_4/save/RestoreV2:15*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_4/save/Assign_16Assign rnn/layer/biases/Variable_1/Adamrnn_4/save/RestoreV2:16*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_4/save/Assign_17Assign"rnn/layer/biases/Variable_1/Adam_1rnn_4/save/RestoreV2:17*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_4/save/Assign_18Assignrnn/layer/weights/Variable/Adamrnn_4/save/RestoreV2:18*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save/Assign_19Assign!rnn/layer/weights/Variable/Adam_1rnn_4/save/RestoreV2:19*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_4/save/Assign_20Assign!rnn/layer/weights/Variable_1/Adamrnn_4/save/RestoreV2:20*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save/Assign_21Assign#rnn/layer/weights/Variable_1/Adam_1rnn_4/save/RestoreV2:21*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_4/save/Assign_22Assign rnn/layer_1/biases/Variable/Adamrnn_4/save/RestoreV2:22*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn_4/save/Assign_23Assign"rnn/layer_1/biases/Variable/Adam_1rnn_4/save/RestoreV2:23*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_4/save/Assign_24Assign"rnn/layer_1/biases/Variable_1/Adamrnn_4/save/RestoreV2:24*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_4/save/Assign_25Assign$rnn/layer_1/biases/Variable_1/Adam_1rnn_4/save/RestoreV2:25*
use_locking(*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_4/save/Assign_26Assign!rnn/layer_1/weights/Variable/Adamrnn_4/save/RestoreV2:26*
use_locking(*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_4/save/Assign_27Assign#rnn/layer_1/weights/Variable/Adam_1rnn_4/save/RestoreV2:27*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save/Assign_28Assign#rnn/layer_1/weights/Variable_1/Adamrnn_4/save/RestoreV2:28*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save/Assign_29Assign%rnn/layer_1/weights/Variable_1/Adam_1rnn_4/save/RestoreV2:29*
use_locking(*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_4/save/Assign_30Assign rnn/layer_2/biases/Variable/Adamrnn_4/save/RestoreV2:30*
use_locking(*
T0**
_class 
loc:@layer_2/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_4/save/Assign_31Assign"rnn/layer_2/biases/Variable/Adam_1rnn_4/save/RestoreV2:31*
T0**
_class 
loc:@layer_2/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn_4/save/Assign_32Assign"rnn/layer_2/biases/Variable_1/Adamrnn_4/save/RestoreV2:32*
use_locking(*
T0*,
_class"
 loc:@layer_2/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_4/save/Assign_33Assign$rnn/layer_2/biases/Variable_1/Adam_1rnn_4/save/RestoreV2:33*
use_locking(*
T0*,
_class"
 loc:@layer_2/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_4/save/Assign_34Assign!rnn/layer_2/weights/Variable/Adamrnn_4/save/RestoreV2:34*
T0*+
_class!
loc:@layer_2/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save/Assign_35Assign#rnn/layer_2/weights/Variable/Adam_1rnn_4/save/RestoreV2:35*
use_locking(*
T0*+
_class!
loc:@layer_2/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_4/save/Assign_36Assign#rnn/layer_2/weights/Variable_1/Adamrnn_4/save/RestoreV2:36*
use_locking(*
T0*-
_class#
!loc:@layer_2/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_4/save/Assign_37Assign%rnn/layer_2/weights/Variable_1/Adam_1rnn_4/save/RestoreV2:37*
T0*-
_class#
!loc:@layer_2/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save/Assign_38Assignrnn/rnn/basic_lstm_cell/biasrnn_4/save/RestoreV2:38*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
rnn_4/save/Assign_39Assignrnn/rnn/basic_lstm_cell/kernelrnn_4/save/RestoreV2:39*
use_locking(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
rnn_4/save/Assign_40Assign%rnn/rnn/rnn/basic_lstm_cell/bias/Adamrnn_4/save/RestoreV2:40*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
rnn_4/save/Assign_41Assign'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1rnn_4/save/RestoreV2:41*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
rnn_4/save/Assign_42Assign'rnn/rnn/rnn/basic_lstm_cell/kernel/Adamrnn_4/save/RestoreV2:42*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
rnn_4/save/Assign_43Assign)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn_4/save/RestoreV2:43*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
rnn_4/save/Assign_44Assignrnn_2/beta1_powerrnn_4/save/RestoreV2:44*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn_4/save/Assign_45Assignrnn_2/beta2_powerrnn_4/save/RestoreV2:45*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(
�
rnn_4/save/Assign_46Assignrnn_4/beta1_powerrnn_4/save/RestoreV2:46*
T0**
_class 
loc:@layer_2/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(
�
rnn_4/save/Assign_47Assignrnn_4/beta2_powerrnn_4/save/RestoreV2:47*
T0**
_class 
loc:@layer_2/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(
�
rnn_4/save/restore_allNoOp^rnn_4/save/Assign^rnn_4/save/Assign_1^rnn_4/save/Assign_10^rnn_4/save/Assign_11^rnn_4/save/Assign_12^rnn_4/save/Assign_13^rnn_4/save/Assign_14^rnn_4/save/Assign_15^rnn_4/save/Assign_16^rnn_4/save/Assign_17^rnn_4/save/Assign_18^rnn_4/save/Assign_19^rnn_4/save/Assign_2^rnn_4/save/Assign_20^rnn_4/save/Assign_21^rnn_4/save/Assign_22^rnn_4/save/Assign_23^rnn_4/save/Assign_24^rnn_4/save/Assign_25^rnn_4/save/Assign_26^rnn_4/save/Assign_27^rnn_4/save/Assign_28^rnn_4/save/Assign_29^rnn_4/save/Assign_3^rnn_4/save/Assign_30^rnn_4/save/Assign_31^rnn_4/save/Assign_32^rnn_4/save/Assign_33^rnn_4/save/Assign_34^rnn_4/save/Assign_35^rnn_4/save/Assign_36^rnn_4/save/Assign_37^rnn_4/save/Assign_38^rnn_4/save/Assign_39^rnn_4/save/Assign_4^rnn_4/save/Assign_40^rnn_4/save/Assign_41^rnn_4/save/Assign_42^rnn_4/save/Assign_43^rnn_4/save/Assign_44^rnn_4/save/Assign_45^rnn_4/save/Assign_46^rnn_4/save/Assign_47^rnn_4/save/Assign_5^rnn_4/save/Assign_6^rnn_4/save/Assign_7^rnn_4/save/Assign_8^rnn_4/save/Assign_9
V
rnn_4/PlaceholderPlaceholder*
dtype0*
_output_shapes
:*
shape:
�

rnn_4/initNoOp^layer/biases/Variable/Assign^layer/biases/Variable_1/Assign^layer/weights/Variable/Assign ^layer/weights/Variable_1/Assign^layer_1/biases/Variable/Assign!^layer_1/biases/Variable_1/Assign ^layer_1/weights/Variable/Assign"^layer_1/weights/Variable_1/Assign^layer_2/biases/Variable/Assign!^layer_2/biases/Variable_1/Assign ^layer_2/weights/Variable/Assign"^layer_2/weights/Variable_1/Assign^rnn/beta1_power/Assign^rnn/beta2_power/Assign&^rnn/layer/biases/Variable/Adam/Assign(^rnn/layer/biases/Variable/Adam_1/Assign(^rnn/layer/biases/Variable_1/Adam/Assign*^rnn/layer/biases/Variable_1/Adam_1/Assign'^rnn/layer/weights/Variable/Adam/Assign)^rnn/layer/weights/Variable/Adam_1/Assign)^rnn/layer/weights/Variable_1/Adam/Assign+^rnn/layer/weights/Variable_1/Adam_1/Assign(^rnn/layer_1/biases/Variable/Adam/Assign*^rnn/layer_1/biases/Variable/Adam_1/Assign*^rnn/layer_1/biases/Variable_1/Adam/Assign,^rnn/layer_1/biases/Variable_1/Adam_1/Assign)^rnn/layer_1/weights/Variable/Adam/Assign+^rnn/layer_1/weights/Variable/Adam_1/Assign+^rnn/layer_1/weights/Variable_1/Adam/Assign-^rnn/layer_1/weights/Variable_1/Adam_1/Assign(^rnn/layer_2/biases/Variable/Adam/Assign*^rnn/layer_2/biases/Variable/Adam_1/Assign*^rnn/layer_2/biases/Variable_1/Adam/Assign,^rnn/layer_2/biases/Variable_1/Adam_1/Assign)^rnn/layer_2/weights/Variable/Adam/Assign+^rnn/layer_2/weights/Variable/Adam_1/Assign+^rnn/layer_2/weights/Variable_1/Adam/Assign-^rnn/layer_2/weights/Variable_1/Adam_1/Assign$^rnn/rnn/basic_lstm_cell/bias/Assign&^rnn/rnn/basic_lstm_cell/kernel/Assign-^rnn/rnn/rnn/basic_lstm_cell/bias/Adam/Assign/^rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1/Assign/^rnn/rnn/rnn/basic_lstm_cell/kernel/Adam/Assign1^rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1/Assign^rnn_2/beta1_power/Assign^rnn_2/beta2_power/Assign^rnn_4/beta1_power/Assign^rnn_4/beta2_power/Assign
X
rnn_4/save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
 rnn_4/save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_a49c8027b1174b35a8d569ce00c839e0/part*
dtype0*
_output_shapes
: 
�
rnn_4/save_1/StringJoin
StringJoinrnn_4/save_1/Const rnn_4/save_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Y
rnn_4/save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
s
"rnn_4/save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
rnn_4/save_1/ShardedFilenameShardedFilenamernn_4/save_1/StringJoin"rnn_4/save_1/ShardedFilename/shardrnn_4/save_1/num_shards"/device:CPU:0*
_output_shapes
: 
�
 rnn_4/save_1/SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�0Blayer/biases/VariableBlayer/biases/Variable_1Blayer/weights/VariableBlayer/weights/Variable_1Blayer_1/biases/VariableBlayer_1/biases/Variable_1Blayer_1/weights/VariableBlayer_1/weights/Variable_1Blayer_2/biases/VariableBlayer_2/biases/Variable_1Blayer_2/weights/VariableBlayer_2/weights/Variable_1Brnn/beta1_powerBrnn/beta2_powerBrnn/layer/biases/Variable/AdamB rnn/layer/biases/Variable/Adam_1B rnn/layer/biases/Variable_1/AdamB"rnn/layer/biases/Variable_1/Adam_1Brnn/layer/weights/Variable/AdamB!rnn/layer/weights/Variable/Adam_1B!rnn/layer/weights/Variable_1/AdamB#rnn/layer/weights/Variable_1/Adam_1B rnn/layer_1/biases/Variable/AdamB"rnn/layer_1/biases/Variable/Adam_1B"rnn/layer_1/biases/Variable_1/AdamB$rnn/layer_1/biases/Variable_1/Adam_1B!rnn/layer_1/weights/Variable/AdamB#rnn/layer_1/weights/Variable/Adam_1B#rnn/layer_1/weights/Variable_1/AdamB%rnn/layer_1/weights/Variable_1/Adam_1B rnn/layer_2/biases/Variable/AdamB"rnn/layer_2/biases/Variable/Adam_1B"rnn/layer_2/biases/Variable_1/AdamB$rnn/layer_2/biases/Variable_1/Adam_1B!rnn/layer_2/weights/Variable/AdamB#rnn/layer_2/weights/Variable/Adam_1B#rnn/layer_2/weights/Variable_1/AdamB%rnn/layer_2/weights/Variable_1/Adam_1Brnn/rnn/basic_lstm_cell/biasBrnn/rnn/basic_lstm_cell/kernelB%rnn/rnn/rnn/basic_lstm_cell/bias/AdamB'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1B'rnn/rnn/rnn/basic_lstm_cell/kernel/AdamB)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1Brnn_2/beta1_powerBrnn_2/beta2_powerBrnn_4/beta1_powerBrnn_4/beta2_power*
dtype0*
_output_shapes
:0
�
$rnn_4/save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:0
�
rnn_4/save_1/SaveV2SaveV2rnn_4/save_1/ShardedFilename rnn_4/save_1/SaveV2/tensor_names$rnn_4/save_1/SaveV2/shape_and_sliceslayer/biases/Variablelayer/biases/Variable_1layer/weights/Variablelayer/weights/Variable_1layer_1/biases/Variablelayer_1/biases/Variable_1layer_1/weights/Variablelayer_1/weights/Variable_1layer_2/biases/Variablelayer_2/biases/Variable_1layer_2/weights/Variablelayer_2/weights/Variable_1rnn/beta1_powerrnn/beta2_powerrnn/layer/biases/Variable/Adam rnn/layer/biases/Variable/Adam_1 rnn/layer/biases/Variable_1/Adam"rnn/layer/biases/Variable_1/Adam_1rnn/layer/weights/Variable/Adam!rnn/layer/weights/Variable/Adam_1!rnn/layer/weights/Variable_1/Adam#rnn/layer/weights/Variable_1/Adam_1 rnn/layer_1/biases/Variable/Adam"rnn/layer_1/biases/Variable/Adam_1"rnn/layer_1/biases/Variable_1/Adam$rnn/layer_1/biases/Variable_1/Adam_1!rnn/layer_1/weights/Variable/Adam#rnn/layer_1/weights/Variable/Adam_1#rnn/layer_1/weights/Variable_1/Adam%rnn/layer_1/weights/Variable_1/Adam_1 rnn/layer_2/biases/Variable/Adam"rnn/layer_2/biases/Variable/Adam_1"rnn/layer_2/biases/Variable_1/Adam$rnn/layer_2/biases/Variable_1/Adam_1!rnn/layer_2/weights/Variable/Adam#rnn/layer_2/weights/Variable/Adam_1#rnn/layer_2/weights/Variable_1/Adam%rnn/layer_2/weights/Variable_1/Adam_1rnn/rnn/basic_lstm_cell/biasrnn/rnn/basic_lstm_cell/kernel%rnn/rnn/rnn/basic_lstm_cell/bias/Adam'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1'rnn/rnn/rnn/basic_lstm_cell/kernel/Adam)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn_2/beta1_powerrnn_2/beta2_powerrnn_4/beta1_powerrnn_4/beta2_power"/device:CPU:0*>
dtypes4
220
�
rnn_4/save_1/control_dependencyIdentityrnn_4/save_1/ShardedFilename^rnn_4/save_1/SaveV2"/device:CPU:0*
T0*/
_class%
#!loc:@rnn_4/save_1/ShardedFilename*
_output_shapes
: 
�
3rnn_4/save_1/MergeV2Checkpoints/checkpoint_prefixesPackrnn_4/save_1/ShardedFilename ^rnn_4/save_1/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
rnn_4/save_1/MergeV2CheckpointsMergeV2Checkpoints3rnn_4/save_1/MergeV2Checkpoints/checkpoint_prefixesrnn_4/save_1/Const"/device:CPU:0*
delete_old_dirs(
�
rnn_4/save_1/IdentityIdentityrnn_4/save_1/Const ^rnn_4/save_1/MergeV2Checkpoints ^rnn_4/save_1/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
�
#rnn_4/save_1/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�0Blayer/biases/VariableBlayer/biases/Variable_1Blayer/weights/VariableBlayer/weights/Variable_1Blayer_1/biases/VariableBlayer_1/biases/Variable_1Blayer_1/weights/VariableBlayer_1/weights/Variable_1Blayer_2/biases/VariableBlayer_2/biases/Variable_1Blayer_2/weights/VariableBlayer_2/weights/Variable_1Brnn/beta1_powerBrnn/beta2_powerBrnn/layer/biases/Variable/AdamB rnn/layer/biases/Variable/Adam_1B rnn/layer/biases/Variable_1/AdamB"rnn/layer/biases/Variable_1/Adam_1Brnn/layer/weights/Variable/AdamB!rnn/layer/weights/Variable/Adam_1B!rnn/layer/weights/Variable_1/AdamB#rnn/layer/weights/Variable_1/Adam_1B rnn/layer_1/biases/Variable/AdamB"rnn/layer_1/biases/Variable/Adam_1B"rnn/layer_1/biases/Variable_1/AdamB$rnn/layer_1/biases/Variable_1/Adam_1B!rnn/layer_1/weights/Variable/AdamB#rnn/layer_1/weights/Variable/Adam_1B#rnn/layer_1/weights/Variable_1/AdamB%rnn/layer_1/weights/Variable_1/Adam_1B rnn/layer_2/biases/Variable/AdamB"rnn/layer_2/biases/Variable/Adam_1B"rnn/layer_2/biases/Variable_1/AdamB$rnn/layer_2/biases/Variable_1/Adam_1B!rnn/layer_2/weights/Variable/AdamB#rnn/layer_2/weights/Variable/Adam_1B#rnn/layer_2/weights/Variable_1/AdamB%rnn/layer_2/weights/Variable_1/Adam_1Brnn/rnn/basic_lstm_cell/biasBrnn/rnn/basic_lstm_cell/kernelB%rnn/rnn/rnn/basic_lstm_cell/bias/AdamB'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1B'rnn/rnn/rnn/basic_lstm_cell/kernel/AdamB)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1Brnn_2/beta1_powerBrnn_2/beta2_powerBrnn_4/beta1_powerBrnn_4/beta2_power*
dtype0*
_output_shapes
:0
�
'rnn_4/save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:0
�
rnn_4/save_1/RestoreV2	RestoreV2rnn_4/save_1/Const#rnn_4/save_1/RestoreV2/tensor_names'rnn_4/save_1/RestoreV2/shape_and_slices"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
220
�
rnn_4/save_1/AssignAssignlayer/biases/Variablernn_4/save_1/RestoreV2*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_4/save_1/Assign_1Assignlayer/biases/Variable_1rnn_4/save_1/RestoreV2:1*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_4/save_1/Assign_2Assignlayer/weights/Variablernn_4/save_1/RestoreV2:2*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_4/save_1/Assign_3Assignlayer/weights/Variable_1rnn_4/save_1/RestoreV2:3*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_4/save_1/Assign_4Assignlayer_1/biases/Variablernn_4/save_1/RestoreV2:4*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_4/save_1/Assign_5Assignlayer_1/biases/Variable_1rnn_4/save_1/RestoreV2:5*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_4/save_1/Assign_6Assignlayer_1/weights/Variablernn_4/save_1/RestoreV2:6*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save_1/Assign_7Assignlayer_1/weights/Variable_1rnn_4/save_1/RestoreV2:7*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save_1/Assign_8Assignlayer_2/biases/Variablernn_4/save_1/RestoreV2:8*
T0**
_class 
loc:@layer_2/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn_4/save_1/Assign_9Assignlayer_2/biases/Variable_1rnn_4/save_1/RestoreV2:9*
T0*,
_class"
 loc:@layer_2/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_4/save_1/Assign_10Assignlayer_2/weights/Variablernn_4/save_1/RestoreV2:10*
T0*+
_class!
loc:@layer_2/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save_1/Assign_11Assignlayer_2/weights/Variable_1rnn_4/save_1/RestoreV2:11*
use_locking(*
T0*-
_class#
!loc:@layer_2/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_4/save_1/Assign_12Assignrnn/beta1_powerrnn_4/save_1/RestoreV2:12*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn_4/save_1/Assign_13Assignrnn/beta2_powerrnn_4/save_1/RestoreV2:13*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(
�
rnn_4/save_1/Assign_14Assignrnn/layer/biases/Variable/Adamrnn_4/save_1/RestoreV2:14*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_4/save_1/Assign_15Assign rnn/layer/biases/Variable/Adam_1rnn_4/save_1/RestoreV2:15*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn_4/save_1/Assign_16Assign rnn/layer/biases/Variable_1/Adamrnn_4/save_1/RestoreV2:16*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_4/save_1/Assign_17Assign"rnn/layer/biases/Variable_1/Adam_1rnn_4/save_1/RestoreV2:17*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_4/save_1/Assign_18Assignrnn/layer/weights/Variable/Adamrnn_4/save_1/RestoreV2:18*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_4/save_1/Assign_19Assign!rnn/layer/weights/Variable/Adam_1rnn_4/save_1/RestoreV2:19*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save_1/Assign_20Assign!rnn/layer/weights/Variable_1/Adamrnn_4/save_1/RestoreV2:20*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save_1/Assign_21Assign#rnn/layer/weights/Variable_1/Adam_1rnn_4/save_1/RestoreV2:21*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save_1/Assign_22Assign rnn/layer_1/biases/Variable/Adamrnn_4/save_1/RestoreV2:22*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:
*
use_locking(
�
rnn_4/save_1/Assign_23Assign"rnn/layer_1/biases/Variable/Adam_1rnn_4/save_1/RestoreV2:23*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_4/save_1/Assign_24Assign"rnn/layer_1/biases/Variable_1/Adamrnn_4/save_1/RestoreV2:24*
use_locking(*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_4/save_1/Assign_25Assign$rnn/layer_1/biases/Variable_1/Adam_1rnn_4/save_1/RestoreV2:25*
T0*,
_class"
 loc:@layer_1/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_4/save_1/Assign_26Assign!rnn/layer_1/weights/Variable/Adamrnn_4/save_1/RestoreV2:26*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save_1/Assign_27Assign#rnn/layer_1/weights/Variable/Adam_1rnn_4/save_1/RestoreV2:27*
use_locking(*
T0*+
_class!
loc:@layer_1/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn_4/save_1/Assign_28Assign#rnn/layer_1/weights/Variable_1/Adamrnn_4/save_1/RestoreV2:28*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save_1/Assign_29Assign%rnn/layer_1/weights/Variable_1/Adam_1rnn_4/save_1/RestoreV2:29*
T0*-
_class#
!loc:@layer_1/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save_1/Assign_30Assign rnn/layer_2/biases/Variable/Adamrnn_4/save_1/RestoreV2:30*
use_locking(*
T0**
_class 
loc:@layer_2/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_4/save_1/Assign_31Assign"rnn/layer_2/biases/Variable/Adam_1rnn_4/save_1/RestoreV2:31*
use_locking(*
T0**
_class 
loc:@layer_2/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn_4/save_1/Assign_32Assign"rnn/layer_2/biases/Variable_1/Adamrnn_4/save_1/RestoreV2:32*
T0*,
_class"
 loc:@layer_2/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn_4/save_1/Assign_33Assign$rnn/layer_2/biases/Variable_1/Adam_1rnn_4/save_1/RestoreV2:33*
use_locking(*
T0*,
_class"
 loc:@layer_2/biases/Variable_1*
validate_shape(*
_output_shapes
:
�
rnn_4/save_1/Assign_34Assign!rnn/layer_2/weights/Variable/Adamrnn_4/save_1/RestoreV2:34*
T0*+
_class!
loc:@layer_2/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save_1/Assign_35Assign#rnn/layer_2/weights/Variable/Adam_1rnn_4/save_1/RestoreV2:35*
T0*+
_class!
loc:@layer_2/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save_1/Assign_36Assign#rnn/layer_2/weights/Variable_1/Adamrnn_4/save_1/RestoreV2:36*
use_locking(*
T0*-
_class#
!loc:@layer_2/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn_4/save_1/Assign_37Assign%rnn/layer_2/weights/Variable_1/Adam_1rnn_4/save_1/RestoreV2:37*
T0*-
_class#
!loc:@layer_2/weights/Variable_1*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn_4/save_1/Assign_38Assignrnn/rnn/basic_lstm_cell/biasrnn_4/save_1/RestoreV2:38*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
rnn_4/save_1/Assign_39Assignrnn/rnn/basic_lstm_cell/kernelrnn_4/save_1/RestoreV2:39*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
rnn_4/save_1/Assign_40Assign%rnn/rnn/rnn/basic_lstm_cell/bias/Adamrnn_4/save_1/RestoreV2:40*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
rnn_4/save_1/Assign_41Assign'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1rnn_4/save_1/RestoreV2:41*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
rnn_4/save_1/Assign_42Assign'rnn/rnn/rnn/basic_lstm_cell/kernel/Adamrnn_4/save_1/RestoreV2:42*
use_locking(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
rnn_4/save_1/Assign_43Assign)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn_4/save_1/RestoreV2:43*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
�
rnn_4/save_1/Assign_44Assignrnn_2/beta1_powerrnn_4/save_1/RestoreV2:44*
use_locking(*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn_4/save_1/Assign_45Assignrnn_2/beta2_powerrnn_4/save_1/RestoreV2:45*
T0**
_class 
loc:@layer_1/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(
�
rnn_4/save_1/Assign_46Assignrnn_4/beta1_powerrnn_4/save_1/RestoreV2:46*
use_locking(*
T0**
_class 
loc:@layer_2/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn_4/save_1/Assign_47Assignrnn_4/beta2_powerrnn_4/save_1/RestoreV2:47*
T0**
_class 
loc:@layer_2/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(
�	
rnn_4/save_1/restore_shardNoOp^rnn_4/save_1/Assign^rnn_4/save_1/Assign_1^rnn_4/save_1/Assign_10^rnn_4/save_1/Assign_11^rnn_4/save_1/Assign_12^rnn_4/save_1/Assign_13^rnn_4/save_1/Assign_14^rnn_4/save_1/Assign_15^rnn_4/save_1/Assign_16^rnn_4/save_1/Assign_17^rnn_4/save_1/Assign_18^rnn_4/save_1/Assign_19^rnn_4/save_1/Assign_2^rnn_4/save_1/Assign_20^rnn_4/save_1/Assign_21^rnn_4/save_1/Assign_22^rnn_4/save_1/Assign_23^rnn_4/save_1/Assign_24^rnn_4/save_1/Assign_25^rnn_4/save_1/Assign_26^rnn_4/save_1/Assign_27^rnn_4/save_1/Assign_28^rnn_4/save_1/Assign_29^rnn_4/save_1/Assign_3^rnn_4/save_1/Assign_30^rnn_4/save_1/Assign_31^rnn_4/save_1/Assign_32^rnn_4/save_1/Assign_33^rnn_4/save_1/Assign_34^rnn_4/save_1/Assign_35^rnn_4/save_1/Assign_36^rnn_4/save_1/Assign_37^rnn_4/save_1/Assign_38^rnn_4/save_1/Assign_39^rnn_4/save_1/Assign_4^rnn_4/save_1/Assign_40^rnn_4/save_1/Assign_41^rnn_4/save_1/Assign_42^rnn_4/save_1/Assign_43^rnn_4/save_1/Assign_44^rnn_4/save_1/Assign_45^rnn_4/save_1/Assign_46^rnn_4/save_1/Assign_47^rnn_4/save_1/Assign_5^rnn_4/save_1/Assign_6^rnn_4/save_1/Assign_7^rnn_4/save_1/Assign_8^rnn_4/save_1/Assign_9
=
rnn_4/save_1/restore_allNoOp^rnn_4/save_1/restore_shard"T
rnn_4/save_1/Const:0rnn_4/save_1/Identity:0rnn_4/save_1/restore_all (5 @F8"�
trainable_variables��
y
layer/weights/Variable:0layer/weights/Variable/Assignlayer/weights/Variable/read:02layer/weights/random_normal:08
�
layer/weights/Variable_1:0layer/weights/Variable_1/Assignlayer/weights/Variable_1/read:02layer/weights/random_normal_1:08
m
layer/biases/Variable:0layer/biases/Variable/Assignlayer/biases/Variable/read:02layer/biases/Const:08
u
layer/biases/Variable_1:0layer/biases/Variable_1/Assignlayer/biases/Variable_1/read:02layer/biases/Const_1:08
�
 rnn/rnn/basic_lstm_cell/kernel:0%rnn/rnn/basic_lstm_cell/kernel/Assign%rnn/rnn/basic_lstm_cell/kernel/read:02;rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform:08
�
rnn/rnn/basic_lstm_cell/bias:0#rnn/rnn/basic_lstm_cell/bias/Assign#rnn/rnn/basic_lstm_cell/bias/read:020rnn/rnn/basic_lstm_cell/bias/Initializer/zeros:08
�
layer_1/weights/Variable:0layer_1/weights/Variable/Assignlayer_1/weights/Variable/read:02layer_1/weights/random_normal:08
�
layer_1/weights/Variable_1:0!layer_1/weights/Variable_1/Assign!layer_1/weights/Variable_1/read:02!layer_1/weights/random_normal_1:08
u
layer_1/biases/Variable:0layer_1/biases/Variable/Assignlayer_1/biases/Variable/read:02layer_1/biases/Const:08
}
layer_1/biases/Variable_1:0 layer_1/biases/Variable_1/Assign layer_1/biases/Variable_1/read:02layer_1/biases/Const_1:08
�
layer_2/weights/Variable:0layer_2/weights/Variable/Assignlayer_2/weights/Variable/read:02layer_2/weights/random_normal:08
�
layer_2/weights/Variable_1:0!layer_2/weights/Variable_1/Assign!layer_2/weights/Variable_1/read:02!layer_2/weights/random_normal_1:08
u
layer_2/biases/Variable:0layer_2/biases/Variable/Assignlayer_2/biases/Variable/read:02layer_2/biases/Const:08
}
layer_2/biases/Variable_1:0 layer_2/biases/Variable_1/Assign layer_2/biases/Variable_1/read:02layer_2/biases/Const_1:08"0
train_op$
"
rnn/Adam

rnn_2/Adam

rnn_4/Adam"��
while_context����
�<
rnn/rnn/while/while_context *rnn/rnn/while/LoopCond:02rnn/rnn/while/Merge:0:rnn/rnn/while/Identity:0Brnn/rnn/while/Exit:0Brnn/rnn/while/Exit_1:0Brnn/rnn/while/Exit_2:0Brnn/rnn/while/Exit_3:0Brnn/rnn/while/Exit_4:0Brnn/gradients/f_count_2:0J�8
rnn/gradients/Add/y:0
rnn/gradients/Add:0
rnn/gradients/Merge:0
rnn/gradients/Merge:1
rnn/gradients/NextIteration:0
rnn/gradients/Switch:0
rnn/gradients/Switch:1
rnn/gradients/f_count:0
rnn/gradients/f_count_1:0
rnn/gradients/f_count_2:0
^rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
drnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
^rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0
Nrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Enter:0
Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2:0
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc:0
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter:0
Jrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2:0
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc:0
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Enter:0
Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2:0
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc:0
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter:0
Jrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2:0
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc:0
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/Enter:0
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2:0
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc:0
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Enter:0
Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2:0
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc:0
?rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/Shape:0
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter:0
Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1:0
Lrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2:0
Nrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1:0
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc:0
Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1:0
rnn/rnn/Minimum:0
rnn/rnn/TensorArray:0
Drnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
rnn/rnn/TensorArray_1:0
#rnn/rnn/basic_lstm_cell/bias/read:0
%rnn/rnn/basic_lstm_cell/kernel/read:0
rnn/rnn/strided_slice_1:0
rnn/rnn/while/Enter:0
rnn/rnn/while/Enter_1:0
rnn/rnn/while/Enter_2:0
rnn/rnn/while/Enter_3:0
rnn/rnn/while/Enter_4:0
rnn/rnn/while/Exit:0
rnn/rnn/while/Exit_1:0
rnn/rnn/while/Exit_2:0
rnn/rnn/while/Exit_3:0
rnn/rnn/while/Exit_4:0
rnn/rnn/while/Identity:0
rnn/rnn/while/Identity_1:0
rnn/rnn/while/Identity_2:0
rnn/rnn/while/Identity_3:0
rnn/rnn/while/Identity_4:0
rnn/rnn/while/Less/Enter:0
rnn/rnn/while/Less:0
rnn/rnn/while/Less_1/Enter:0
rnn/rnn/while/Less_1:0
rnn/rnn/while/LogicalAnd:0
rnn/rnn/while/LoopCond:0
rnn/rnn/while/Merge:0
rnn/rnn/while/Merge:1
rnn/rnn/while/Merge_1:0
rnn/rnn/while/Merge_1:1
rnn/rnn/while/Merge_2:0
rnn/rnn/while/Merge_2:1
rnn/rnn/while/Merge_3:0
rnn/rnn/while/Merge_3:1
rnn/rnn/while/Merge_4:0
rnn/rnn/while/Merge_4:1
rnn/rnn/while/NextIteration:0
rnn/rnn/while/NextIteration_1:0
rnn/rnn/while/NextIteration_2:0
rnn/rnn/while/NextIteration_3:0
rnn/rnn/while/NextIteration_4:0
rnn/rnn/while/Switch:0
rnn/rnn/while/Switch:1
rnn/rnn/while/Switch_1:0
rnn/rnn/while/Switch_1:1
rnn/rnn/while/Switch_2:0
rnn/rnn/while/Switch_2:1
rnn/rnn/while/Switch_3:0
rnn/rnn/while/Switch_3:1
rnn/rnn/while/Switch_4:0
rnn/rnn/while/Switch_4:1
'rnn/rnn/while/TensorArrayReadV3/Enter:0
)rnn/rnn/while/TensorArrayReadV3/Enter_1:0
!rnn/rnn/while/TensorArrayReadV3:0
9rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
3rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
rnn/rnn/while/add/y:0
rnn/rnn/while/add:0
rnn/rnn/while/add_1/y:0
rnn/rnn/while/add_1:0
#rnn/rnn/while/basic_lstm_cell/Add:0
%rnn/rnn/while/basic_lstm_cell/Add_1:0
-rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter:0
'rnn/rnn/while/basic_lstm_cell/BiasAdd:0
%rnn/rnn/while/basic_lstm_cell/Const:0
'rnn/rnn/while/basic_lstm_cell/Const_1:0
'rnn/rnn/while/basic_lstm_cell/Const_2:0
,rnn/rnn/while/basic_lstm_cell/MatMul/Enter:0
&rnn/rnn/while/basic_lstm_cell/MatMul:0
#rnn/rnn/while/basic_lstm_cell/Mul:0
%rnn/rnn/while/basic_lstm_cell/Mul_1:0
%rnn/rnn/while/basic_lstm_cell/Mul_2:0
'rnn/rnn/while/basic_lstm_cell/Sigmoid:0
)rnn/rnn/while/basic_lstm_cell/Sigmoid_1:0
)rnn/rnn/while/basic_lstm_cell/Sigmoid_2:0
$rnn/rnn/while/basic_lstm_cell/Tanh:0
&rnn/rnn/while/basic_lstm_cell/Tanh_1:0
+rnn/rnn/while/basic_lstm_cell/concat/axis:0
&rnn/rnn/while/basic_lstm_cell/concat:0
%rnn/rnn/while/basic_lstm_cell/split:0
%rnn/rnn/while/basic_lstm_cell/split:1
%rnn/rnn/while/basic_lstm_cell/split:2
%rnn/rnn/while/basic_lstm_cell/split:3T
#rnn/rnn/basic_lstm_cell/bias/read:0-rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter:0�
Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0R
rnn/rnn/TensorArray:09rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0�
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc:0Drnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter:01
rnn/rnn/Minimum:0rnn/rnn/while/Less_1/Enter:0�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc:0Frnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter:0�
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc:0Drnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter:0�
^rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0^rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0�
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc:0Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Enter:0�
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc:0Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Enter:07
rnn/rnn/strided_slice_1:0rnn/rnn/while/Less/Enter:0B
rnn/rnn/TensorArray_1:0'rnn/rnn/while/TensorArrayReadV3/Enter:0�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc:0@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/Enter:0�
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc:0Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Enter:0�
Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1:0Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1:0U
%rnn/rnn/basic_lstm_cell/kernel/read:0,rnn/rnn/while/basic_lstm_cell/MatMul/Enter:0q
Drnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0)rnn/rnn/while/TensorArrayReadV3/Enter_1:0Rrnn/rnn/while/Enter:0Rrnn/rnn/while/Enter_1:0Rrnn/rnn/while/Enter_2:0Rrnn/rnn/while/Enter_3:0Rrnn/rnn/while/Enter_4:0Rrnn/gradients/f_count_1:0Zrnn/rnn/strided_slice_1:0
�
rnn_1/rnn/while/while_context *rnn_1/rnn/while/LoopCond:02rnn_1/rnn/while/Merge:0:rnn_1/rnn/while/Identity:0Brnn_1/rnn/while/Exit:0Brnn_1/rnn/while/Exit_1:0Brnn_1/rnn/while/Exit_2:0Brnn_1/rnn/while/Exit_3:0Brnn_1/rnn/while/Exit_4:0J�
#rnn/rnn/basic_lstm_cell/bias/read:0
%rnn/rnn/basic_lstm_cell/kernel/read:0
rnn_1/rnn/Minimum:0
rnn_1/rnn/TensorArray:0
Frnn_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
rnn_1/rnn/TensorArray_1:0
rnn_1/rnn/strided_slice_1:0
rnn_1/rnn/while/Enter:0
rnn_1/rnn/while/Enter_1:0
rnn_1/rnn/while/Enter_2:0
rnn_1/rnn/while/Enter_3:0
rnn_1/rnn/while/Enter_4:0
rnn_1/rnn/while/Exit:0
rnn_1/rnn/while/Exit_1:0
rnn_1/rnn/while/Exit_2:0
rnn_1/rnn/while/Exit_3:0
rnn_1/rnn/while/Exit_4:0
rnn_1/rnn/while/Identity:0
rnn_1/rnn/while/Identity_1:0
rnn_1/rnn/while/Identity_2:0
rnn_1/rnn/while/Identity_3:0
rnn_1/rnn/while/Identity_4:0
rnn_1/rnn/while/Less/Enter:0
rnn_1/rnn/while/Less:0
rnn_1/rnn/while/Less_1/Enter:0
rnn_1/rnn/while/Less_1:0
rnn_1/rnn/while/LogicalAnd:0
rnn_1/rnn/while/LoopCond:0
rnn_1/rnn/while/Merge:0
rnn_1/rnn/while/Merge:1
rnn_1/rnn/while/Merge_1:0
rnn_1/rnn/while/Merge_1:1
rnn_1/rnn/while/Merge_2:0
rnn_1/rnn/while/Merge_2:1
rnn_1/rnn/while/Merge_3:0
rnn_1/rnn/while/Merge_3:1
rnn_1/rnn/while/Merge_4:0
rnn_1/rnn/while/Merge_4:1
rnn_1/rnn/while/NextIteration:0
!rnn_1/rnn/while/NextIteration_1:0
!rnn_1/rnn/while/NextIteration_2:0
!rnn_1/rnn/while/NextIteration_3:0
!rnn_1/rnn/while/NextIteration_4:0
rnn_1/rnn/while/Switch:0
rnn_1/rnn/while/Switch:1
rnn_1/rnn/while/Switch_1:0
rnn_1/rnn/while/Switch_1:1
rnn_1/rnn/while/Switch_2:0
rnn_1/rnn/while/Switch_2:1
rnn_1/rnn/while/Switch_3:0
rnn_1/rnn/while/Switch_3:1
rnn_1/rnn/while/Switch_4:0
rnn_1/rnn/while/Switch_4:1
)rnn_1/rnn/while/TensorArrayReadV3/Enter:0
+rnn_1/rnn/while/TensorArrayReadV3/Enter_1:0
#rnn_1/rnn/while/TensorArrayReadV3:0
;rnn_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
5rnn_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
rnn_1/rnn/while/add/y:0
rnn_1/rnn/while/add:0
rnn_1/rnn/while/add_1/y:0
rnn_1/rnn/while/add_1:0
%rnn_1/rnn/while/basic_lstm_cell/Add:0
'rnn_1/rnn/while/basic_lstm_cell/Add_1:0
/rnn_1/rnn/while/basic_lstm_cell/BiasAdd/Enter:0
)rnn_1/rnn/while/basic_lstm_cell/BiasAdd:0
'rnn_1/rnn/while/basic_lstm_cell/Const:0
)rnn_1/rnn/while/basic_lstm_cell/Const_1:0
)rnn_1/rnn/while/basic_lstm_cell/Const_2:0
.rnn_1/rnn/while/basic_lstm_cell/MatMul/Enter:0
(rnn_1/rnn/while/basic_lstm_cell/MatMul:0
%rnn_1/rnn/while/basic_lstm_cell/Mul:0
'rnn_1/rnn/while/basic_lstm_cell/Mul_1:0
'rnn_1/rnn/while/basic_lstm_cell/Mul_2:0
)rnn_1/rnn/while/basic_lstm_cell/Sigmoid:0
+rnn_1/rnn/while/basic_lstm_cell/Sigmoid_1:0
+rnn_1/rnn/while/basic_lstm_cell/Sigmoid_2:0
&rnn_1/rnn/while/basic_lstm_cell/Tanh:0
(rnn_1/rnn/while/basic_lstm_cell/Tanh_1:0
-rnn_1/rnn/while/basic_lstm_cell/concat/axis:0
(rnn_1/rnn/while/basic_lstm_cell/concat:0
'rnn_1/rnn/while/basic_lstm_cell/split:0
'rnn_1/rnn/while/basic_lstm_cell/split:1
'rnn_1/rnn/while/basic_lstm_cell/split:2
'rnn_1/rnn/while/basic_lstm_cell/split:3u
Frnn_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0+rnn_1/rnn/while/TensorArrayReadV3/Enter_1:0F
rnn_1/rnn/TensorArray_1:0)rnn_1/rnn/while/TensorArrayReadV3/Enter:0;
rnn_1/rnn/strided_slice_1:0rnn_1/rnn/while/Less/Enter:0V
#rnn/rnn/basic_lstm_cell/bias/read:0/rnn_1/rnn/while/basic_lstm_cell/BiasAdd/Enter:0V
rnn_1/rnn/TensorArray:0;rnn_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:05
rnn_1/rnn/Minimum:0rnn_1/rnn/while/Less_1/Enter:0W
%rnn/rnn/basic_lstm_cell/kernel/read:0.rnn_1/rnn/while/basic_lstm_cell/MatMul/Enter:0Rrnn_1/rnn/while/Enter:0Rrnn_1/rnn/while/Enter_1:0Rrnn_1/rnn/while/Enter_2:0Rrnn_1/rnn/while/Enter_3:0Rrnn_1/rnn/while/Enter_4:0Zrnn_1/rnn/strided_slice_1:0
�?
rnn_2/rnn/while/while_context *rnn_2/rnn/while/LoopCond:02rnn_2/rnn/while/Merge:0:rnn_2/rnn/while/Identity:0Brnn_2/rnn/while/Exit:0Brnn_2/rnn/while/Exit_1:0Brnn_2/rnn/while/Exit_2:0Brnn_2/rnn/while/Exit_3:0Brnn_2/rnn/while/Exit_4:0Brnn_2/gradients/f_count_2:0J�<
#rnn/rnn/basic_lstm_cell/bias/read:0
%rnn/rnn/basic_lstm_cell/kernel/read:0
rnn_2/gradients/Add/y:0
rnn_2/gradients/Add:0
rnn_2/gradients/Merge:0
rnn_2/gradients/Merge:1
rnn_2/gradients/NextIteration:0
rnn_2/gradients/Switch:0
rnn_2/gradients/Switch:1
rnn_2/gradients/f_count:0
rnn_2/gradients/f_count_1:0
rnn_2/gradients/f_count_2:0
brnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
hrnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
brnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
Lrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0
Rrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Lrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0
Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Enter:0
Lrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2:0
Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc:0
Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter:0
Nrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2:0
Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc:0
Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Enter:0
Lrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2:0
Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc:0
Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter:0
Nrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2:0
Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc:0
Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/Enter:0
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2:0
Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc:0
Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Enter:0
Lrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2:0
Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc:0
Crnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/Shape:0
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter:0
Lrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1:0
Prnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2:0
Rrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1:0
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc:0
Lrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1:0
rnn_2/rnn/Minimum:0
rnn_2/rnn/TensorArray:0
Frnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
rnn_2/rnn/TensorArray_1:0
rnn_2/rnn/strided_slice_1:0
rnn_2/rnn/while/Enter:0
rnn_2/rnn/while/Enter_1:0
rnn_2/rnn/while/Enter_2:0
rnn_2/rnn/while/Enter_3:0
rnn_2/rnn/while/Enter_4:0
rnn_2/rnn/while/Exit:0
rnn_2/rnn/while/Exit_1:0
rnn_2/rnn/while/Exit_2:0
rnn_2/rnn/while/Exit_3:0
rnn_2/rnn/while/Exit_4:0
rnn_2/rnn/while/Identity:0
rnn_2/rnn/while/Identity_1:0
rnn_2/rnn/while/Identity_2:0
rnn_2/rnn/while/Identity_3:0
rnn_2/rnn/while/Identity_4:0
rnn_2/rnn/while/Less/Enter:0
rnn_2/rnn/while/Less:0
rnn_2/rnn/while/Less_1/Enter:0
rnn_2/rnn/while/Less_1:0
rnn_2/rnn/while/LogicalAnd:0
rnn_2/rnn/while/LoopCond:0
rnn_2/rnn/while/Merge:0
rnn_2/rnn/while/Merge:1
rnn_2/rnn/while/Merge_1:0
rnn_2/rnn/while/Merge_1:1
rnn_2/rnn/while/Merge_2:0
rnn_2/rnn/while/Merge_2:1
rnn_2/rnn/while/Merge_3:0
rnn_2/rnn/while/Merge_3:1
rnn_2/rnn/while/Merge_4:0
rnn_2/rnn/while/Merge_4:1
rnn_2/rnn/while/NextIteration:0
!rnn_2/rnn/while/NextIteration_1:0
!rnn_2/rnn/while/NextIteration_2:0
!rnn_2/rnn/while/NextIteration_3:0
!rnn_2/rnn/while/NextIteration_4:0
rnn_2/rnn/while/Switch:0
rnn_2/rnn/while/Switch:1
rnn_2/rnn/while/Switch_1:0
rnn_2/rnn/while/Switch_1:1
rnn_2/rnn/while/Switch_2:0
rnn_2/rnn/while/Switch_2:1
rnn_2/rnn/while/Switch_3:0
rnn_2/rnn/while/Switch_3:1
rnn_2/rnn/while/Switch_4:0
rnn_2/rnn/while/Switch_4:1
)rnn_2/rnn/while/TensorArrayReadV3/Enter:0
+rnn_2/rnn/while/TensorArrayReadV3/Enter_1:0
#rnn_2/rnn/while/TensorArrayReadV3:0
;rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
5rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
rnn_2/rnn/while/add/y:0
rnn_2/rnn/while/add:0
rnn_2/rnn/while/add_1/y:0
rnn_2/rnn/while/add_1:0
%rnn_2/rnn/while/basic_lstm_cell/Add:0
'rnn_2/rnn/while/basic_lstm_cell/Add_1:0
/rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter:0
)rnn_2/rnn/while/basic_lstm_cell/BiasAdd:0
'rnn_2/rnn/while/basic_lstm_cell/Const:0
)rnn_2/rnn/while/basic_lstm_cell/Const_1:0
)rnn_2/rnn/while/basic_lstm_cell/Const_2:0
.rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter:0
(rnn_2/rnn/while/basic_lstm_cell/MatMul:0
%rnn_2/rnn/while/basic_lstm_cell/Mul:0
'rnn_2/rnn/while/basic_lstm_cell/Mul_1:0
'rnn_2/rnn/while/basic_lstm_cell/Mul_2:0
)rnn_2/rnn/while/basic_lstm_cell/Sigmoid:0
+rnn_2/rnn/while/basic_lstm_cell/Sigmoid_1:0
+rnn_2/rnn/while/basic_lstm_cell/Sigmoid_2:0
&rnn_2/rnn/while/basic_lstm_cell/Tanh:0
(rnn_2/rnn/while/basic_lstm_cell/Tanh_1:0
-rnn_2/rnn/while/basic_lstm_cell/concat/axis:0
(rnn_2/rnn/while/basic_lstm_cell/concat:0
'rnn_2/rnn/while/basic_lstm_cell/split:0
'rnn_2/rnn/while/basic_lstm_cell/split:1
'rnn_2/rnn/while/basic_lstm_cell/split:2
'rnn_2/rnn/while/basic_lstm_cell/split:3�
Lrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0Lrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:05
rnn_2/rnn/Minimum:0rnn_2/rnn/while/Less_1/Enter:0�
Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc:0Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Enter:0u
Frnn_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0+rnn_2/rnn/while/TensorArrayReadV3/Enter_1:0�
Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc:0Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter:0�
Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc:0Hrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter:0W
%rnn/rnn/basic_lstm_cell/kernel/read:0.rnn_2/rnn/while/basic_lstm_cell/MatMul/Enter:0;
rnn_2/rnn/strided_slice_1:0rnn_2/rnn/while/Less/Enter:0V
rnn_2/rnn/TensorArray:0;rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0�
Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc:0Drnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul/Enter:0F
rnn_2/rnn/TensorArray_1:0)rnn_2/rnn/while/TensorArrayReadV3/Enter:0�
Lrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1:0Lrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1:0V
#rnn/rnn/basic_lstm_cell/bias/read:0/rnn_2/rnn/while/basic_lstm_cell/BiasAdd/Enter:0�
Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc:0Jrnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter:0�
brnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0brnn_2/gradients/rnn_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0�
Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc:0Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Enter:0�
Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc:0Frnn_2/gradients/rnn_2/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Enter:0Rrnn_2/rnn/while/Enter:0Rrnn_2/rnn/while/Enter_1:0Rrnn_2/rnn/while/Enter_2:0Rrnn_2/rnn/while/Enter_3:0Rrnn_2/rnn/while/Enter_4:0Rrnn_2/gradients/f_count_1:0Zrnn_2/rnn/strided_slice_1:0
�
rnn_3/rnn/while/while_context *rnn_3/rnn/while/LoopCond:02rnn_3/rnn/while/Merge:0:rnn_3/rnn/while/Identity:0Brnn_3/rnn/while/Exit:0Brnn_3/rnn/while/Exit_1:0Brnn_3/rnn/while/Exit_2:0Brnn_3/rnn/while/Exit_3:0Brnn_3/rnn/while/Exit_4:0J�
#rnn/rnn/basic_lstm_cell/bias/read:0
%rnn/rnn/basic_lstm_cell/kernel/read:0
rnn_3/rnn/Minimum:0
rnn_3/rnn/TensorArray:0
Frnn_3/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
rnn_3/rnn/TensorArray_1:0
rnn_3/rnn/strided_slice_1:0
rnn_3/rnn/while/Enter:0
rnn_3/rnn/while/Enter_1:0
rnn_3/rnn/while/Enter_2:0
rnn_3/rnn/while/Enter_3:0
rnn_3/rnn/while/Enter_4:0
rnn_3/rnn/while/Exit:0
rnn_3/rnn/while/Exit_1:0
rnn_3/rnn/while/Exit_2:0
rnn_3/rnn/while/Exit_3:0
rnn_3/rnn/while/Exit_4:0
rnn_3/rnn/while/Identity:0
rnn_3/rnn/while/Identity_1:0
rnn_3/rnn/while/Identity_2:0
rnn_3/rnn/while/Identity_3:0
rnn_3/rnn/while/Identity_4:0
rnn_3/rnn/while/Less/Enter:0
rnn_3/rnn/while/Less:0
rnn_3/rnn/while/Less_1/Enter:0
rnn_3/rnn/while/Less_1:0
rnn_3/rnn/while/LogicalAnd:0
rnn_3/rnn/while/LoopCond:0
rnn_3/rnn/while/Merge:0
rnn_3/rnn/while/Merge:1
rnn_3/rnn/while/Merge_1:0
rnn_3/rnn/while/Merge_1:1
rnn_3/rnn/while/Merge_2:0
rnn_3/rnn/while/Merge_2:1
rnn_3/rnn/while/Merge_3:0
rnn_3/rnn/while/Merge_3:1
rnn_3/rnn/while/Merge_4:0
rnn_3/rnn/while/Merge_4:1
rnn_3/rnn/while/NextIteration:0
!rnn_3/rnn/while/NextIteration_1:0
!rnn_3/rnn/while/NextIteration_2:0
!rnn_3/rnn/while/NextIteration_3:0
!rnn_3/rnn/while/NextIteration_4:0
rnn_3/rnn/while/Switch:0
rnn_3/rnn/while/Switch:1
rnn_3/rnn/while/Switch_1:0
rnn_3/rnn/while/Switch_1:1
rnn_3/rnn/while/Switch_2:0
rnn_3/rnn/while/Switch_2:1
rnn_3/rnn/while/Switch_3:0
rnn_3/rnn/while/Switch_3:1
rnn_3/rnn/while/Switch_4:0
rnn_3/rnn/while/Switch_4:1
)rnn_3/rnn/while/TensorArrayReadV3/Enter:0
+rnn_3/rnn/while/TensorArrayReadV3/Enter_1:0
#rnn_3/rnn/while/TensorArrayReadV3:0
;rnn_3/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
5rnn_3/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
rnn_3/rnn/while/add/y:0
rnn_3/rnn/while/add:0
rnn_3/rnn/while/add_1/y:0
rnn_3/rnn/while/add_1:0
%rnn_3/rnn/while/basic_lstm_cell/Add:0
'rnn_3/rnn/while/basic_lstm_cell/Add_1:0
/rnn_3/rnn/while/basic_lstm_cell/BiasAdd/Enter:0
)rnn_3/rnn/while/basic_lstm_cell/BiasAdd:0
'rnn_3/rnn/while/basic_lstm_cell/Const:0
)rnn_3/rnn/while/basic_lstm_cell/Const_1:0
)rnn_3/rnn/while/basic_lstm_cell/Const_2:0
.rnn_3/rnn/while/basic_lstm_cell/MatMul/Enter:0
(rnn_3/rnn/while/basic_lstm_cell/MatMul:0
%rnn_3/rnn/while/basic_lstm_cell/Mul:0
'rnn_3/rnn/while/basic_lstm_cell/Mul_1:0
'rnn_3/rnn/while/basic_lstm_cell/Mul_2:0
)rnn_3/rnn/while/basic_lstm_cell/Sigmoid:0
+rnn_3/rnn/while/basic_lstm_cell/Sigmoid_1:0
+rnn_3/rnn/while/basic_lstm_cell/Sigmoid_2:0
&rnn_3/rnn/while/basic_lstm_cell/Tanh:0
(rnn_3/rnn/while/basic_lstm_cell/Tanh_1:0
-rnn_3/rnn/while/basic_lstm_cell/concat/axis:0
(rnn_3/rnn/while/basic_lstm_cell/concat:0
'rnn_3/rnn/while/basic_lstm_cell/split:0
'rnn_3/rnn/while/basic_lstm_cell/split:1
'rnn_3/rnn/while/basic_lstm_cell/split:2
'rnn_3/rnn/while/basic_lstm_cell/split:3u
Frnn_3/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0+rnn_3/rnn/while/TensorArrayReadV3/Enter_1:0V
rnn_3/rnn/TensorArray:0;rnn_3/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:05
rnn_3/rnn/Minimum:0rnn_3/rnn/while/Less_1/Enter:0W
%rnn/rnn/basic_lstm_cell/kernel/read:0.rnn_3/rnn/while/basic_lstm_cell/MatMul/Enter:0F
rnn_3/rnn/TensorArray_1:0)rnn_3/rnn/while/TensorArrayReadV3/Enter:0;
rnn_3/rnn/strided_slice_1:0rnn_3/rnn/while/Less/Enter:0V
#rnn/rnn/basic_lstm_cell/bias/read:0/rnn_3/rnn/while/basic_lstm_cell/BiasAdd/Enter:0Rrnn_3/rnn/while/Enter:0Rrnn_3/rnn/while/Enter_1:0Rrnn_3/rnn/while/Enter_2:0Rrnn_3/rnn/while/Enter_3:0Rrnn_3/rnn/while/Enter_4:0Zrnn_3/rnn/strided_slice_1:0
�?
rnn_4/rnn/while/while_context *rnn_4/rnn/while/LoopCond:02rnn_4/rnn/while/Merge:0:rnn_4/rnn/while/Identity:0Brnn_4/rnn/while/Exit:0Brnn_4/rnn/while/Exit_1:0Brnn_4/rnn/while/Exit_2:0Brnn_4/rnn/while/Exit_3:0Brnn_4/rnn/while/Exit_4:0Brnn_4/gradients/f_count_2:0J�<
#rnn/rnn/basic_lstm_cell/bias/read:0
%rnn/rnn/basic_lstm_cell/kernel/read:0
rnn_4/gradients/Add/y:0
rnn_4/gradients/Add:0
rnn_4/gradients/Merge:0
rnn_4/gradients/Merge:1
rnn_4/gradients/NextIteration:0
rnn_4/gradients/Switch:0
rnn_4/gradients/Switch:1
rnn_4/gradients/f_count:0
rnn_4/gradients/f_count_1:0
rnn_4/gradients/f_count_2:0
brnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
hrnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
brnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
Lrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0
Rrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Lrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0
Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Enter:0
Lrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2:0
Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc:0
Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter:0
Nrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2:0
Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc:0
Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Enter:0
Lrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2:0
Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc:0
Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter:0
Nrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2:0
Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc:0
Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/Enter:0
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2:0
Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc:0
Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Enter:0
Lrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2:0
Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc:0
Crnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/Shape:0
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter:0
Lrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1:0
Prnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2:0
Rrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1:0
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc:0
Lrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1:0
rnn_4/rnn/Minimum:0
rnn_4/rnn/TensorArray:0
Frnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
rnn_4/rnn/TensorArray_1:0
rnn_4/rnn/strided_slice_1:0
rnn_4/rnn/while/Enter:0
rnn_4/rnn/while/Enter_1:0
rnn_4/rnn/while/Enter_2:0
rnn_4/rnn/while/Enter_3:0
rnn_4/rnn/while/Enter_4:0
rnn_4/rnn/while/Exit:0
rnn_4/rnn/while/Exit_1:0
rnn_4/rnn/while/Exit_2:0
rnn_4/rnn/while/Exit_3:0
rnn_4/rnn/while/Exit_4:0
rnn_4/rnn/while/Identity:0
rnn_4/rnn/while/Identity_1:0
rnn_4/rnn/while/Identity_2:0
rnn_4/rnn/while/Identity_3:0
rnn_4/rnn/while/Identity_4:0
rnn_4/rnn/while/Less/Enter:0
rnn_4/rnn/while/Less:0
rnn_4/rnn/while/Less_1/Enter:0
rnn_4/rnn/while/Less_1:0
rnn_4/rnn/while/LogicalAnd:0
rnn_4/rnn/while/LoopCond:0
rnn_4/rnn/while/Merge:0
rnn_4/rnn/while/Merge:1
rnn_4/rnn/while/Merge_1:0
rnn_4/rnn/while/Merge_1:1
rnn_4/rnn/while/Merge_2:0
rnn_4/rnn/while/Merge_2:1
rnn_4/rnn/while/Merge_3:0
rnn_4/rnn/while/Merge_3:1
rnn_4/rnn/while/Merge_4:0
rnn_4/rnn/while/Merge_4:1
rnn_4/rnn/while/NextIteration:0
!rnn_4/rnn/while/NextIteration_1:0
!rnn_4/rnn/while/NextIteration_2:0
!rnn_4/rnn/while/NextIteration_3:0
!rnn_4/rnn/while/NextIteration_4:0
rnn_4/rnn/while/Switch:0
rnn_4/rnn/while/Switch:1
rnn_4/rnn/while/Switch_1:0
rnn_4/rnn/while/Switch_1:1
rnn_4/rnn/while/Switch_2:0
rnn_4/rnn/while/Switch_2:1
rnn_4/rnn/while/Switch_3:0
rnn_4/rnn/while/Switch_3:1
rnn_4/rnn/while/Switch_4:0
rnn_4/rnn/while/Switch_4:1
)rnn_4/rnn/while/TensorArrayReadV3/Enter:0
+rnn_4/rnn/while/TensorArrayReadV3/Enter_1:0
#rnn_4/rnn/while/TensorArrayReadV3:0
;rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
5rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
rnn_4/rnn/while/add/y:0
rnn_4/rnn/while/add:0
rnn_4/rnn/while/add_1/y:0
rnn_4/rnn/while/add_1:0
%rnn_4/rnn/while/basic_lstm_cell/Add:0
'rnn_4/rnn/while/basic_lstm_cell/Add_1:0
/rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter:0
)rnn_4/rnn/while/basic_lstm_cell/BiasAdd:0
'rnn_4/rnn/while/basic_lstm_cell/Const:0
)rnn_4/rnn/while/basic_lstm_cell/Const_1:0
)rnn_4/rnn/while/basic_lstm_cell/Const_2:0
.rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter:0
(rnn_4/rnn/while/basic_lstm_cell/MatMul:0
%rnn_4/rnn/while/basic_lstm_cell/Mul:0
'rnn_4/rnn/while/basic_lstm_cell/Mul_1:0
'rnn_4/rnn/while/basic_lstm_cell/Mul_2:0
)rnn_4/rnn/while/basic_lstm_cell/Sigmoid:0
+rnn_4/rnn/while/basic_lstm_cell/Sigmoid_1:0
+rnn_4/rnn/while/basic_lstm_cell/Sigmoid_2:0
&rnn_4/rnn/while/basic_lstm_cell/Tanh:0
(rnn_4/rnn/while/basic_lstm_cell/Tanh_1:0
-rnn_4/rnn/while/basic_lstm_cell/concat/axis:0
(rnn_4/rnn/while/basic_lstm_cell/concat:0
'rnn_4/rnn/while/basic_lstm_cell/split:0
'rnn_4/rnn/while/basic_lstm_cell/split:1
'rnn_4/rnn/while/basic_lstm_cell/split:2
'rnn_4/rnn/while/basic_lstm_cell/split:3�
Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc:0Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter:0;
rnn_4/rnn/strided_slice_1:0rnn_4/rnn/while/Less/Enter:0V
rnn_4/rnn/TensorArray:0;rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0�
Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc:0Drnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul/Enter:0�
Lrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1:0Lrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1:0W
%rnn/rnn/basic_lstm_cell/kernel/read:0.rnn_4/rnn/while/basic_lstm_cell/MatMul/Enter:0�
Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc:0Jrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter:0u
Frnn_4/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0+rnn_4/rnn/while/TensorArrayReadV3/Enter_1:0�
brnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0brnn_4/gradients/rnn_4/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0�
Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc:0Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Enter:0�
Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc:0Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Enter:05
rnn_4/rnn/Minimum:0rnn_4/rnn/while/Less_1/Enter:0F
rnn_4/rnn/TensorArray_1:0)rnn_4/rnn/while/TensorArrayReadV3/Enter:0�
Lrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0Lrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0V
#rnn/rnn/basic_lstm_cell/bias/read:0/rnn_4/rnn/while/basic_lstm_cell/BiasAdd/Enter:0�
Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc:0Frnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Enter:0�
Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc:0Hrnn_4/gradients/rnn_4/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter:0Rrnn_4/rnn/while/Enter:0Rrnn_4/rnn/while/Enter_1:0Rrnn_4/rnn/while/Enter_2:0Rrnn_4/rnn/while/Enter_3:0Rrnn_4/rnn/while/Enter_4:0Rrnn_4/gradients/f_count_1:0Zrnn_4/rnn/strided_slice_1:0"�<
	variables�<�<
y
layer/weights/Variable:0layer/weights/Variable/Assignlayer/weights/Variable/read:02layer/weights/random_normal:08
�
layer/weights/Variable_1:0layer/weights/Variable_1/Assignlayer/weights/Variable_1/read:02layer/weights/random_normal_1:08
m
layer/biases/Variable:0layer/biases/Variable/Assignlayer/biases/Variable/read:02layer/biases/Const:08
u
layer/biases/Variable_1:0layer/biases/Variable_1/Assignlayer/biases/Variable_1/read:02layer/biases/Const_1:08
�
 rnn/rnn/basic_lstm_cell/kernel:0%rnn/rnn/basic_lstm_cell/kernel/Assign%rnn/rnn/basic_lstm_cell/kernel/read:02;rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform:08
�
rnn/rnn/basic_lstm_cell/bias:0#rnn/rnn/basic_lstm_cell/bias/Assign#rnn/rnn/basic_lstm_cell/bias/read:020rnn/rnn/basic_lstm_cell/bias/Initializer/zeros:08
d
rnn/beta1_power:0rnn/beta1_power/Assignrnn/beta1_power/read:02rnn/beta1_power/initial_value:0
d
rnn/beta2_power:0rnn/beta2_power/Assignrnn/beta2_power/read:02rnn/beta2_power/initial_value:0
�
!rnn/layer/weights/Variable/Adam:0&rnn/layer/weights/Variable/Adam/Assign&rnn/layer/weights/Variable/Adam/read:023rnn/layer/weights/Variable/Adam/Initializer/zeros:0
�
#rnn/layer/weights/Variable/Adam_1:0(rnn/layer/weights/Variable/Adam_1/Assign(rnn/layer/weights/Variable/Adam_1/read:025rnn/layer/weights/Variable/Adam_1/Initializer/zeros:0
�
#rnn/layer/weights/Variable_1/Adam:0(rnn/layer/weights/Variable_1/Adam/Assign(rnn/layer/weights/Variable_1/Adam/read:025rnn/layer/weights/Variable_1/Adam/Initializer/zeros:0
�
%rnn/layer/weights/Variable_1/Adam_1:0*rnn/layer/weights/Variable_1/Adam_1/Assign*rnn/layer/weights/Variable_1/Adam_1/read:027rnn/layer/weights/Variable_1/Adam_1/Initializer/zeros:0
�
 rnn/layer/biases/Variable/Adam:0%rnn/layer/biases/Variable/Adam/Assign%rnn/layer/biases/Variable/Adam/read:022rnn/layer/biases/Variable/Adam/Initializer/zeros:0
�
"rnn/layer/biases/Variable/Adam_1:0'rnn/layer/biases/Variable/Adam_1/Assign'rnn/layer/biases/Variable/Adam_1/read:024rnn/layer/biases/Variable/Adam_1/Initializer/zeros:0
�
"rnn/layer/biases/Variable_1/Adam:0'rnn/layer/biases/Variable_1/Adam/Assign'rnn/layer/biases/Variable_1/Adam/read:024rnn/layer/biases/Variable_1/Adam/Initializer/zeros:0
�
$rnn/layer/biases/Variable_1/Adam_1:0)rnn/layer/biases/Variable_1/Adam_1/Assign)rnn/layer/biases/Variable_1/Adam_1/read:026rnn/layer/biases/Variable_1/Adam_1/Initializer/zeros:0
�
)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam:0.rnn/rnn/rnn/basic_lstm_cell/kernel/Adam/Assign.rnn/rnn/rnn/basic_lstm_cell/kernel/Adam/read:02;rnn/rnn/rnn/basic_lstm_cell/kernel/Adam/Initializer/zeros:0
�
+rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1:00rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1/Assign0rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1/read:02=rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros:0
�
'rnn/rnn/rnn/basic_lstm_cell/bias/Adam:0,rnn/rnn/rnn/basic_lstm_cell/bias/Adam/Assign,rnn/rnn/rnn/basic_lstm_cell/bias/Adam/read:029rnn/rnn/rnn/basic_lstm_cell/bias/Adam/Initializer/zeros:0
�
)rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1:0.rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1/Assign.rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1/read:02;rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1/Initializer/zeros:0
�
layer_1/weights/Variable:0layer_1/weights/Variable/Assignlayer_1/weights/Variable/read:02layer_1/weights/random_normal:08
�
layer_1/weights/Variable_1:0!layer_1/weights/Variable_1/Assign!layer_1/weights/Variable_1/read:02!layer_1/weights/random_normal_1:08
u
layer_1/biases/Variable:0layer_1/biases/Variable/Assignlayer_1/biases/Variable/read:02layer_1/biases/Const:08
}
layer_1/biases/Variable_1:0 layer_1/biases/Variable_1/Assign layer_1/biases/Variable_1/read:02layer_1/biases/Const_1:08
l
rnn_2/beta1_power:0rnn_2/beta1_power/Assignrnn_2/beta1_power/read:02!rnn_2/beta1_power/initial_value:0
l
rnn_2/beta2_power:0rnn_2/beta2_power/Assignrnn_2/beta2_power/read:02!rnn_2/beta2_power/initial_value:0
�
#rnn/layer_1/weights/Variable/Adam:0(rnn/layer_1/weights/Variable/Adam/Assign(rnn/layer_1/weights/Variable/Adam/read:025rnn/layer_1/weights/Variable/Adam/Initializer/zeros:0
�
%rnn/layer_1/weights/Variable/Adam_1:0*rnn/layer_1/weights/Variable/Adam_1/Assign*rnn/layer_1/weights/Variable/Adam_1/read:027rnn/layer_1/weights/Variable/Adam_1/Initializer/zeros:0
�
%rnn/layer_1/weights/Variable_1/Adam:0*rnn/layer_1/weights/Variable_1/Adam/Assign*rnn/layer_1/weights/Variable_1/Adam/read:027rnn/layer_1/weights/Variable_1/Adam/Initializer/zeros:0
�
'rnn/layer_1/weights/Variable_1/Adam_1:0,rnn/layer_1/weights/Variable_1/Adam_1/Assign,rnn/layer_1/weights/Variable_1/Adam_1/read:029rnn/layer_1/weights/Variable_1/Adam_1/Initializer/zeros:0
�
"rnn/layer_1/biases/Variable/Adam:0'rnn/layer_1/biases/Variable/Adam/Assign'rnn/layer_1/biases/Variable/Adam/read:024rnn/layer_1/biases/Variable/Adam/Initializer/zeros:0
�
$rnn/layer_1/biases/Variable/Adam_1:0)rnn/layer_1/biases/Variable/Adam_1/Assign)rnn/layer_1/biases/Variable/Adam_1/read:026rnn/layer_1/biases/Variable/Adam_1/Initializer/zeros:0
�
$rnn/layer_1/biases/Variable_1/Adam:0)rnn/layer_1/biases/Variable_1/Adam/Assign)rnn/layer_1/biases/Variable_1/Adam/read:026rnn/layer_1/biases/Variable_1/Adam/Initializer/zeros:0
�
&rnn/layer_1/biases/Variable_1/Adam_1:0+rnn/layer_1/biases/Variable_1/Adam_1/Assign+rnn/layer_1/biases/Variable_1/Adam_1/read:028rnn/layer_1/biases/Variable_1/Adam_1/Initializer/zeros:0
�
layer_2/weights/Variable:0layer_2/weights/Variable/Assignlayer_2/weights/Variable/read:02layer_2/weights/random_normal:08
�
layer_2/weights/Variable_1:0!layer_2/weights/Variable_1/Assign!layer_2/weights/Variable_1/read:02!layer_2/weights/random_normal_1:08
u
layer_2/biases/Variable:0layer_2/biases/Variable/Assignlayer_2/biases/Variable/read:02layer_2/biases/Const:08
}
layer_2/biases/Variable_1:0 layer_2/biases/Variable_1/Assign layer_2/biases/Variable_1/read:02layer_2/biases/Const_1:08
l
rnn_4/beta1_power:0rnn_4/beta1_power/Assignrnn_4/beta1_power/read:02!rnn_4/beta1_power/initial_value:0
l
rnn_4/beta2_power:0rnn_4/beta2_power/Assignrnn_4/beta2_power/read:02!rnn_4/beta2_power/initial_value:0
�
#rnn/layer_2/weights/Variable/Adam:0(rnn/layer_2/weights/Variable/Adam/Assign(rnn/layer_2/weights/Variable/Adam/read:025rnn/layer_2/weights/Variable/Adam/Initializer/zeros:0
�
%rnn/layer_2/weights/Variable/Adam_1:0*rnn/layer_2/weights/Variable/Adam_1/Assign*rnn/layer_2/weights/Variable/Adam_1/read:027rnn/layer_2/weights/Variable/Adam_1/Initializer/zeros:0
�
%rnn/layer_2/weights/Variable_1/Adam:0*rnn/layer_2/weights/Variable_1/Adam/Assign*rnn/layer_2/weights/Variable_1/Adam/read:027rnn/layer_2/weights/Variable_1/Adam/Initializer/zeros:0
�
'rnn/layer_2/weights/Variable_1/Adam_1:0,rnn/layer_2/weights/Variable_1/Adam_1/Assign,rnn/layer_2/weights/Variable_1/Adam_1/read:029rnn/layer_2/weights/Variable_1/Adam_1/Initializer/zeros:0
�
"rnn/layer_2/biases/Variable/Adam:0'rnn/layer_2/biases/Variable/Adam/Assign'rnn/layer_2/biases/Variable/Adam/read:024rnn/layer_2/biases/Variable/Adam/Initializer/zeros:0
�
$rnn/layer_2/biases/Variable/Adam_1:0)rnn/layer_2/biases/Variable/Adam_1/Assign)rnn/layer_2/biases/Variable/Adam_1/read:026rnn/layer_2/biases/Variable/Adam_1/Initializer/zeros:0
�
$rnn/layer_2/biases/Variable_1/Adam:0)rnn/layer_2/biases/Variable_1/Adam/Assign)rnn/layer_2/biases/Variable_1/Adam/read:026rnn/layer_2/biases/Variable_1/Adam/Initializer/zeros:0
�
&rnn/layer_2/biases/Variable_1/Adam_1:0+rnn/layer_2/biases/Variable_1/Adam_1/Assign+rnn/layer_2/biases/Variable_1/Adam_1/read:028rnn/layer_2/biases/Variable_1/Adam_1/Initializer/zeros:0