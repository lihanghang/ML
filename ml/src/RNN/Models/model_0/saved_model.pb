��
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
b'unknown'��
q
inputsPlaceholder*
dtype0*+
_output_shapes
:���������* 
shape:���������
r
outputsPlaceholder*
dtype0*+
_output_shapes
:���������* 
shape:���������
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
layer/weights/random_normalAddlayer/weights/random_normal/mul layer/weights/random_normal/mean*
_output_shapes

:
*
T0
�
layer/weights/Variable
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
layer/weights/Variable/AssignAssignlayer/weights/Variablelayer/weights/random_normal*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
layer/weights/Variable/readIdentitylayer/weights/Variable*)
_class
loc:@layer/weights/Variable*
_output_shapes

:
*
T0
t
#layer/weights/random_normal_1/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
g
"layer/weights/random_normal_1/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
i
$layer/weights/random_normal_1/stddevConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
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
_output_shapes

:
*
	container *
shape
:
*
shared_name *
dtype0
�
layer/weights/Variable_1/AssignAssignlayer/weights/Variable_1layer/weights/random_normal_1*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
layer/weights/Variable_1/readIdentitylayer/weights/Variable_1*
_output_shapes

:
*
T0*+
_class!
loc:@layer/weights/Variable_1
_
layer/biases/ConstConst*
valueB
*���=*
dtype0*
_output_shapes
:

�
layer/biases/Variable
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
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
�
layer/biases/Variable_1/AssignAssignlayer/biases/Variable_1layer/biases/Const_1*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
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

rnn/MatMulMatMulrnn/Reshapelayer/weights/Variable/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
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
valueB:*
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
!rnn/BasicLSTMCellZeroState/concatConcatV2 rnn/BasicLSTMCellZeroState/Const"rnn/BasicLSTMCellZeroState/Const_1&rnn/BasicLSTMCellZeroState/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
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

:

l
"rnn/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
l
"rnn/BasicLSTMCellZeroState/Const_3Const*
dtype0*
_output_shapes
:*
valueB:

l
"rnn/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
l
"rnn/BasicLSTMCellZeroState/Const_5Const*
_output_shapes
:*
valueB:
*
dtype0
j
(rnn/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
#rnn/BasicLSTMCellZeroState/concat_1ConcatV2"rnn/BasicLSTMCellZeroState/Const_4"rnn/BasicLSTMCellZeroState/Const_5(rnn/BasicLSTMCellZeroState/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
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

:

l
"rnn/BasicLSTMCellZeroState/Const_6Const*
dtype0*
_output_shapes
:*
valueB:
l
"rnn/BasicLSTMCellZeroState/Const_7Const*
_output_shapes
:*
valueB:
*
dtype0
N
rnn/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
U
rnn/rnn/range/startConst*
dtype0*
_output_shapes
: *
value	B :
U
rnn/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
v
rnn/rnn/rangeRangernn/rnn/range/startrnn/rnn/Rankrnn/rnn/range/delta*
_output_shapes
:*

Tidx0
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
rnn/rnn/concatConcatV2rnn/rnn/concat/values_0rnn/rnn/rangernn/rnn/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
rnn/rnn/transpose	Transposernn/Reshape_1rnn/rnn/concat*
T0*+
_output_shapes
:���������
*
Tperm0
^
rnn/rnn/ShapeShapernn/rnn/transpose*
out_type0*
_output_shapes
:*
T0
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
shrink_axis_mask*

begin_mask *
ellipsis_mask *
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
rnn/rnn/strided_slice_1StridedSlicernn/rnn/Shape_1rnn/rnn/strided_slice_1/stackrnn/rnn/strided_slice_1/stack_1rnn/rnn/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
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
rnn/rnn/strided_slice_2StridedSlicernn/rnn/Shape_2rnn/rnn/strided_slice_2/stackrnn/rnn/strided_slice_2/stack_1rnn/rnn/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
X
rnn/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn/rnn/ExpandDims
ExpandDimsrnn/rnn/strided_slice_2rnn/rnn/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
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
rnn/rnn/concat_1ConcatV2rnn/rnn/ExpandDimsrnn/rnn/Constrnn/rnn/concat_1/axis*
_output_shapes
:*

Tidx0*
T0*
N
X
rnn/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
rnn/rnn/zerosFillrnn/rnn/concat_1rnn/rnn/zeros/Const*'
_output_shapes
:���������
*
T0*

index_type0
N
rnn/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
�
rnn/rnn/TensorArrayTensorArrayV3rnn/rnn/strided_slice_1*
identical_element_shapes(*3
tensor_array_namernn/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *$
element_shape:���������
*
dynamic_size( *
clear_after_read(
�
rnn/rnn/TensorArray_1TensorArrayV3rnn/rnn/strided_slice_1*2
tensor_array_namernn/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *$
element_shape:���������
*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
q
 rnn/rnn/TensorArrayUnstack/ShapeShapernn/rnn/transpose*
T0*
out_type0*
_output_shapes
:
x
.rnn/rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
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
(rnn/rnn/TensorArrayUnstack/strided_sliceStridedSlice rnn/rnn/TensorArrayUnstack/Shape.rnn/rnn/TensorArrayUnstack/strided_slice/stack0rnn/rnn/TensorArrayUnstack/strided_slice/stack_10rnn/rnn/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
h
&rnn/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
h
&rnn/rnn/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
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
rnn/rnn/MaximumMaximumrnn/rnn/Maximum/xrnn/rnn/strided_slice_1*
_output_shapes
: *
T0
e
rnn/rnn/MinimumMinimumrnn/rnn/strided_slice_1rnn/rnn/Maximum*
_output_shapes
: *
T0
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
rnn/rnn/while/Enter_1Enterrnn/rnn/time*
_output_shapes
: *+

frame_namernn/rnn/while/while_context*
T0*
is_constant( *
parallel_iterations 
�
rnn/rnn/while/Enter_2Enterrnn/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *+

frame_namernn/rnn/while/while_context
�
rnn/rnn/while/Enter_3Enter rnn/BasicLSTMCellZeroState/zeros*
is_constant( *
parallel_iterations *
_output_shapes

:
*+

frame_namernn/rnn/while/while_context*
T0
�
rnn/rnn/while/Enter_4Enter"rnn/BasicLSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
*+

frame_namernn/rnn/while/while_context
z
rnn/rnn/while/MergeMergernn/rnn/while/Enterrnn/rnn/while/NextIteration*
_output_shapes
: : *
T0*
N
�
rnn/rnn/while/Merge_1Mergernn/rnn/while/Enter_1rnn/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
rnn/rnn/while/Merge_2Mergernn/rnn/while/Enter_2rnn/rnn/while/NextIteration_2*
N*
_output_shapes
: : *
T0
�
rnn/rnn/while/Merge_3Mergernn/rnn/while/Enter_3rnn/rnn/while/NextIteration_3*
T0*
N* 
_output_shapes
:
: 
�
rnn/rnn/while/Merge_4Mergernn/rnn/while/Enter_4rnn/rnn/while/NextIteration_4*
T0*
N* 
_output_shapes
:
: 
j
rnn/rnn/while/LessLessrnn/rnn/while/Mergernn/rnn/while/Less/Enter*
T0*
_output_shapes
: 
�
rnn/rnn/while/Less/EnterEnterrnn/rnn/strided_slice_1*
_output_shapes
: *+

frame_namernn/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
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
:
:

�
rnn/rnn/while/Switch_4Switchrnn/rnn/while/Merge_4rnn/rnn/while/LoopCond*(
_class
loc:@rnn/rnn/while/Merge_4*(
_output_shapes
:
:
*
T0
[
rnn/rnn/while/IdentityIdentityrnn/rnn/while/Switch:1*
_output_shapes
: *
T0
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

:

g
rnn/rnn/while/Identity_4Identityrnn/rnn/while/Switch_4:1*
_output_shapes

:
*
T0
n
rnn/rnn/while/add/yConst^rnn/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
f
rnn/rnn/while/addAddrnn/rnn/while/Identityrnn/rnn/while/add/y*
_output_shapes
: *
T0
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
=rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/subSub=rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/max=rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel
�
=rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulGrnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniform=rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
_output_shapes

:(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel
�
9rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniformAdd=rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mul=rnn/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes

:(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel
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
#rnn/rnn/basic_lstm_cell/kernel/readIdentityrnn/rnn/basic_lstm_cell/kernel*
_output_shapes

:(*
T0
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
VariableV2*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
	container *
shape:(*
dtype0*
_output_shapes
:(*
shared_name 
�
#rnn/rnn/basic_lstm_cell/bias/AssignAssignrnn/rnn/basic_lstm_cell/bias.rnn/rnn/basic_lstm_cell/bias/Initializer/zeros*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
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
$rnn/rnn/while/basic_lstm_cell/concatConcatV2rnn/rnn/while/TensorArrayReadV3rnn/rnn/while/Identity_4)rnn/rnn/while/basic_lstm_cell/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
�
$rnn/rnn/while/basic_lstm_cell/MatMulMatMul$rnn/rnn/while/basic_lstm_cell/concat*rnn/rnn/while/basic_lstm_cell/MatMul/Enter*
_output_shapes

:(*
transpose_a( *
transpose_b( *
T0
�
*rnn/rnn/while/basic_lstm_cell/MatMul/EnterEnter#rnn/rnn/basic_lstm_cell/kernel/read*
is_constant(*
parallel_iterations *
_output_shapes

:(*+

frame_namernn/rnn/while/while_context*
T0
�
%rnn/rnn/while/basic_lstm_cell/BiasAddBiasAdd$rnn/rnn/while/basic_lstm_cell/MatMul+rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
_output_shapes

:(*
T0
�
+rnn/rnn/while/basic_lstm_cell/BiasAdd/EnterEnter!rnn/rnn/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:(*+

frame_namernn/rnn/while/while_context
�
%rnn/rnn/while/basic_lstm_cell/Const_1Const^rnn/rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
�
#rnn/rnn/while/basic_lstm_cell/splitSplit#rnn/rnn/while/basic_lstm_cell/Const%rnn/rnn/while/basic_lstm_cell/BiasAdd*<
_output_shapes*
(:
:
:
:
*
	num_split*
T0
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

:

|
%rnn/rnn/while/basic_lstm_cell/SigmoidSigmoid!rnn/rnn/while/basic_lstm_cell/Add*
_output_shapes

:
*
T0
�
!rnn/rnn/while/basic_lstm_cell/MulMulrnn/rnn/while/Identity_3%rnn/rnn/while/basic_lstm_cell/Sigmoid*
T0*
_output_shapes

:

�
'rnn/rnn/while/basic_lstm_cell/Sigmoid_1Sigmoid#rnn/rnn/while/basic_lstm_cell/split*
_output_shapes

:
*
T0
z
"rnn/rnn/while/basic_lstm_cell/TanhTanh%rnn/rnn/while/basic_lstm_cell/split:1*
_output_shapes

:
*
T0
�
#rnn/rnn/while/basic_lstm_cell/Mul_1Mul'rnn/rnn/while/basic_lstm_cell/Sigmoid_1"rnn/rnn/while/basic_lstm_cell/Tanh*
T0*
_output_shapes

:

�
#rnn/rnn/while/basic_lstm_cell/Add_1Add!rnn/rnn/while/basic_lstm_cell/Mul#rnn/rnn/while/basic_lstm_cell/Mul_1*
_output_shapes

:
*
T0
z
$rnn/rnn/while/basic_lstm_cell/Tanh_1Tanh#rnn/rnn/while/basic_lstm_cell/Add_1*
T0*
_output_shapes

:

�
'rnn/rnn/while/basic_lstm_cell/Sigmoid_2Sigmoid%rnn/rnn/while/basic_lstm_cell/split:3*
T0*
_output_shapes

:

�
#rnn/rnn/while/basic_lstm_cell/Mul_2Mul$rnn/rnn/while/basic_lstm_cell/Tanh_1'rnn/rnn/while/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes

:

�
1rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV37rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn/rnn/while/Identity_1#rnn/rnn/while/basic_lstm_cell/Mul_2rnn/rnn/while/Identity_2*
_output_shapes
: *
T0*6
_class,
*(loc:@rnn/rnn/while/basic_lstm_cell/Mul_2
�
7rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn/rnn/TensorArray*
is_constant(*
_output_shapes
:*+

frame_namernn/rnn/while/while_context*
T0*6
_class,
*(loc:@rnn/rnn/while/basic_lstm_cell/Mul_2*
parallel_iterations 
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
rnn/rnn/while/NextIterationNextIterationrnn/rnn/while/add*
_output_shapes
: *
T0
d
rnn/rnn/while/NextIteration_1NextIterationrnn/rnn/while/add_1*
T0*
_output_shapes
: 
�
rnn/rnn/while/NextIteration_2NextIteration1rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
|
rnn/rnn/while/NextIteration_3NextIteration#rnn/rnn/while/basic_lstm_cell/Add_1*
T0*
_output_shapes

:

|
rnn/rnn/while/NextIteration_4NextIteration#rnn/rnn/while/basic_lstm_cell/Mul_2*
_output_shapes

:
*
T0
Q
rnn/rnn/while/ExitExitrnn/rnn/while/Switch*
T0*
_output_shapes
: 
U
rnn/rnn/while/Exit_1Exitrnn/rnn/while/Switch_1*
_output_shapes
: *
T0
U
rnn/rnn/while/Exit_2Exitrnn/rnn/while/Switch_2*
_output_shapes
: *
T0
]
rnn/rnn/while/Exit_3Exitrnn/rnn/while/Switch_3*
_output_shapes

:
*
T0
]
rnn/rnn/while/Exit_4Exitrnn/rnn/while/Switch_4*
T0*
_output_shapes

:

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
rnn/rnn/TensorArrayStack/rangeRange$rnn/rnn/TensorArrayStack/range/start*rnn/rnn/TensorArrayStack/TensorArraySizeV3$rnn/rnn/TensorArrayStack/range/delta*#
_output_shapes
:���������*

Tidx0*&
_class
loc:@rnn/rnn/TensorArray
�
,rnn/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn/rnn/TensorArrayrnn/rnn/TensorArrayStack/rangernn/rnn/while/Exit_2*
element_shape
:
*&
_class
loc:@rnn/rnn/TensorArray*
dtype0*"
_output_shapes
:

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
rnn/rnn/range_1/deltaConst*
_output_shapes
: *
value	B :*
dtype0
~
rnn/rnn/range_1Rangernn/rnn/range_1/startrnn/rnn/Rank_1rnn/rnn/range_1/delta*
_output_shapes
:*

Tidx0
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
rnn/rnn/concat_2ConcatV2rnn/rnn/concat_2/values_0rnn/rnn/range_1rnn/rnn/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
rnn/rnn/transpose_1	Transpose,rnn/rnn/TensorArrayStack/TensorArrayGatherV3rnn/rnn/concat_2*
Tperm0*
T0*"
_output_shapes
:

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

:

�
rnn/MatMul_1MatMulrnn/Reshape_2layer/weights/Variable_1/read*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
e
	rnn/predsAddrnn/MatMul_1layer/biases/Variable_1/read*
T0*
_output_shapes

:
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
:
f
rnn/Reshape_4/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
r
rnn/Reshape_4Reshapeoutputsrnn/Reshape_4/shape*
Tshape0*#
_output_shapes
:���������*
T0
Q
rnn/subSubrnn/Reshape_3rnn/Reshape_4*
_output_shapes
:*
T0
B

rnn/SquareSquarernn/sub*
_output_shapes
:*
T0
S
	rnn/ConstConst*
valueB: *
dtype0*
_output_shapes
:
e
rnn/MeanMean
rnn/Square	rnn/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
V
rnn/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
\
rnn/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
rnn/gradients/MergeMergernn/gradients/f_count_1rnn/gradients/NextIteration*
_output_shapes
: : *
T0*
N
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
rnn/gradients/AddAddrnn/gradients/Switch:1rnn/gradients/Add/y*
_output_shapes
: *
T0
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
rnn/gradients/Switch_1Switchrnn/gradients/Merge_1rnn/gradients/b_count_2*
_output_shapes
: : *
T0
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
)rnn/gradients/rnn/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
#rnn/gradients/rnn/Mean_grad/ReshapeReshapernn/gradients/Fill)rnn/gradients/rnn/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
k
!rnn/gradients/rnn/Mean_grad/ConstConst*
valueB:*
dtype0*
_output_shapes
:
�
 rnn/gradients/rnn/Mean_grad/TileTile#rnn/gradients/rnn/Mean_grad/Reshape!rnn/gradients/rnn/Mean_grad/Const*
_output_shapes
:*

Tmultiples0*
T0
h
#rnn/gradients/rnn/Mean_grad/Const_1Const*
valueB
 *  �A*
dtype0*
_output_shapes
: 
�
#rnn/gradients/rnn/Mean_grad/truedivRealDiv rnn/gradients/rnn/Mean_grad/Tile#rnn/gradients/rnn/Mean_grad/Const_1*
T0*
_output_shapes
:
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
:
�
#rnn/gradients/rnn/Square_grad/Mul_1Mul#rnn/gradients/rnn/Mean_grad/truediv!rnn/gradients/rnn/Square_grad/Mul*
T0*
_output_shapes
:
j
 rnn/gradients/rnn/sub_grad/ShapeConst*
valueB:*
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
rnn/gradients/rnn/sub_grad/SumSum#rnn/gradients/rnn/Square_grad/Mul_10rnn/gradients/rnn/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
"rnn/gradients/rnn/sub_grad/ReshapeReshapernn/gradients/rnn/sub_grad/Sum rnn/gradients/rnn/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
�
 rnn/gradients/rnn/sub_grad/Sum_1Sum#rnn/gradients/rnn/Square_grad/Mul_12rnn/gradients/rnn/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
3rnn/gradients/rnn/sub_grad/tuple/control_dependencyIdentity"rnn/gradients/rnn/sub_grad/Reshape,^rnn/gradients/rnn/sub_grad/tuple/group_deps*5
_class+
)'loc:@rnn/gradients/rnn/sub_grad/Reshape*
_output_shapes
:*
T0
�
5rnn/gradients/rnn/sub_grad/tuple/control_dependency_1Identity$rnn/gradients/rnn/sub_grad/Reshape_1,^rnn/gradients/rnn/sub_grad/tuple/group_deps*7
_class-
+)loc:@rnn/gradients/rnn/sub_grad/Reshape_1*#
_output_shapes
:���������*
T0
w
&rnn/gradients/rnn/Reshape_3_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
(rnn/gradients/rnn/Reshape_3_grad/ReshapeReshape3rnn/gradients/rnn/sub_grad/tuple/control_dependency&rnn/gradients/rnn/Reshape_3_grad/Shape*
T0*
Tshape0*
_output_shapes

:
s
"rnn/gradients/rnn/preds_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
n
$rnn/gradients/rnn/preds_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
2rnn/gradients/rnn/preds_grad/BroadcastGradientArgsBroadcastGradientArgs"rnn/gradients/rnn/preds_grad/Shape$rnn/gradients/rnn/preds_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
 rnn/gradients/rnn/preds_grad/SumSum(rnn/gradients/rnn/Reshape_3_grad/Reshape2rnn/gradients/rnn/preds_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
$rnn/gradients/rnn/preds_grad/ReshapeReshape rnn/gradients/rnn/preds_grad/Sum"rnn/gradients/rnn/preds_grad/Shape*
T0*
Tshape0*
_output_shapes

:
�
"rnn/gradients/rnn/preds_grad/Sum_1Sum(rnn/gradients/rnn/Reshape_3_grad/Reshape4rnn/gradients/rnn/preds_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
&rnn/gradients/rnn/preds_grad/Reshape_1Reshape"rnn/gradients/rnn/preds_grad/Sum_1$rnn/gradients/rnn/preds_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
�
-rnn/gradients/rnn/preds_grad/tuple/group_depsNoOp%^rnn/gradients/rnn/preds_grad/Reshape'^rnn/gradients/rnn/preds_grad/Reshape_1
�
5rnn/gradients/rnn/preds_grad/tuple/control_dependencyIdentity$rnn/gradients/rnn/preds_grad/Reshape.^rnn/gradients/rnn/preds_grad/tuple/group_deps*7
_class-
+)loc:@rnn/gradients/rnn/preds_grad/Reshape*
_output_shapes

:*
T0
�
7rnn/gradients/rnn/preds_grad/tuple/control_dependency_1Identity&rnn/gradients/rnn/preds_grad/Reshape_1.^rnn/gradients/rnn/preds_grad/tuple/group_deps*
T0*9
_class/
-+loc:@rnn/gradients/rnn/preds_grad/Reshape_1*
_output_shapes
:
�
&rnn/gradients/rnn/MatMul_1_grad/MatMulMatMul5rnn/gradients/rnn/preds_grad/tuple/control_dependencylayer/weights/Variable_1/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b(
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

:

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
valueB"      
   *
dtype0*
_output_shapes
:
�
(rnn/gradients/rnn/Reshape_2_grad/ReshapeReshape8rnn/gradients/rnn/MatMul_1_grad/tuple/control_dependency&rnn/gradients/rnn/Reshape_2_grad/Shape*
T0*
Tshape0*"
_output_shapes
:

�
8rnn/gradients/rnn/rnn/transpose_1_grad/InvertPermutationInvertPermutationrnn/rnn/concat_2*
T0*
_output_shapes
:
�
0rnn/gradients/rnn/rnn/transpose_1_grad/transpose	Transpose(rnn/gradients/rnn/Reshape_2_grad/Reshape8rnn/gradients/rnn/rnn/transpose_1_grad/InvertPermutation*
T0*"
_output_shapes
:
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
grnn/gradients/rnn/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3arnn/gradients/rnn/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3rnn/rnn/TensorArrayStack/range0rnn/gradients/rnn/rnn/transpose_1_grad/transpose]rnn/gradients/rnn/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
h
rnn/gradients/zerosConst*
valueB
*    *
dtype0*
_output_shapes

:

j
rnn/gradients/zeros_1Const*
_output_shapes

:
*
valueB
*    *
dtype0
�
.rnn/gradients/rnn/rnn/while/Exit_2_grad/b_exitEntergrnn/gradients/rnn/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *9

frame_name+)rnn/gradients/rnn/rnn/while/while_context
�
.rnn/gradients/rnn/rnn/while/Exit_3_grad/b_exitEnterrnn/gradients/zeros*
_output_shapes

:
*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context*
T0*
is_constant( *
parallel_iterations 
�
.rnn/gradients/rnn/rnn/while/Exit_4_grad/b_exitEnterrnn/gradients/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes

:
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
N* 
_output_shapes
:
: *
T0
�
2rnn/gradients/rnn/rnn/while/Switch_4_grad/b_switchMerge.rnn/gradients/rnn/rnn/while/Exit_4_grad/b_exit9rnn/gradients/rnn/rnn/while/Switch_4_grad_1/NextIteration*
T0*
N* 
_output_shapes
:
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
Crnn/gradients/rnn/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity1rnn/gradients/rnn/rnn/while/Merge_2_grad/Switch:1:^rnn/gradients/rnn/rnn/while/Merge_2_grad/tuple/group_deps*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: *
T0
�
/rnn/gradients/rnn/rnn/while/Merge_3_grad/SwitchSwitch2rnn/gradients/rnn/rnn/while/Switch_3_grad/b_switchrnn/gradients/b_count_2*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_3_grad/b_switch*(
_output_shapes
:
:

s
9rnn/gradients/rnn/rnn/while/Merge_3_grad/tuple/group_depsNoOp0^rnn/gradients/rnn/rnn/while/Merge_3_grad/Switch
�
Arnn/gradients/rnn/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity/rnn/gradients/rnn/rnn/while/Merge_3_grad/Switch:^rnn/gradients/rnn/rnn/while/Merge_3_grad/tuple/group_deps*
_output_shapes

:
*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_3_grad/b_switch
�
Crnn/gradients/rnn/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity1rnn/gradients/rnn/rnn/while/Merge_3_grad/Switch:1:^rnn/gradients/rnn/rnn/while/Merge_3_grad/tuple/group_deps*
_output_shapes

:
*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_3_grad/b_switch
�
/rnn/gradients/rnn/rnn/while/Merge_4_grad/SwitchSwitch2rnn/gradients/rnn/rnn/while/Switch_4_grad/b_switchrnn/gradients/b_count_2*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_4_grad/b_switch*(
_output_shapes
:
:

s
9rnn/gradients/rnn/rnn/while/Merge_4_grad/tuple/group_depsNoOp0^rnn/gradients/rnn/rnn/while/Merge_4_grad/Switch
�
Arnn/gradients/rnn/rnn/while/Merge_4_grad/tuple/control_dependencyIdentity/rnn/gradients/rnn/rnn/while/Merge_4_grad/Switch:^rnn/gradients/rnn/rnn/while/Merge_4_grad/tuple/group_deps*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_4_grad/b_switch*
_output_shapes

:

�
Crnn/gradients/rnn/rnn/while/Merge_4_grad/tuple/control_dependency_1Identity1rnn/gradients/rnn/rnn/while/Merge_4_grad/Switch:1:^rnn/gradients/rnn/rnn/while/Merge_4_grad/tuple/group_deps*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_4_grad/b_switch*
_output_shapes

:

�
-rnn/gradients/rnn/rnn/while/Enter_2_grad/ExitExitArnn/gradients/rnn/rnn/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes
: 
�
-rnn/gradients/rnn/rnn/while/Enter_3_grad/ExitExitArnn/gradients/rnn/rnn/while/Merge_3_grad/tuple/control_dependency*
_output_shapes

:
*
T0
�
-rnn/gradients/rnn/rnn/while/Enter_4_grad/ExitExitArnn/gradients/rnn/rnn/while/Merge_4_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
frnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lrnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterCrnn/gradients/rnn/rnn/while/Merge_2_grad/tuple/control_dependency_1*
_output_shapes

:: *6
_class,
*(loc:@rnn/rnn/while/basic_lstm_cell/Mul_2*
sourcernn/gradients
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
brnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityCrnn/gradients/rnn/rnn/while/Merge_2_grad/tuple/control_dependency_1g^rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*6
_class,
*(loc:@rnn/rnn/while/basic_lstm_cell/Mul_2
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
\rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2\rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*

stack_name *
_output_shapes
:*
	elem_type0*+
_class!
loc:@rnn/rnn/while/Identity_1
�
\rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnter\rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
_output_shapes
:*+

frame_namernn/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
�
brnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2\rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterrnn/rnn/while/Identity_1^rnn/gradients/Add*
_output_shapes
: *
swap_memory( *
T0
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

:

�
_rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityCrnn/gradients/rnn/rnn/while/Merge_2_grad/tuple/control_dependency_1V^rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
�
rnn/gradients/AddNAddNCrnn/gradients/rnn/rnn/while/Merge_4_grad/tuple/control_dependency_1]rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
N*
_output_shapes

:
*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_4_grad/b_switch
�
:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/MulMulrnn/gradients/AddNErnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2*
T0*
_output_shapes

:

�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *:
_class0
.,loc:@rnn/rnn/while/basic_lstm_cell/Sigmoid_2*
valueB :
���������
�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_accStackV2@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Const*

stack_name *
_output_shapes
:*
	elem_type0*:
_class0
.,loc:@rnn/rnn/while/basic_lstm_cell/Sigmoid_2
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

:
*
swap_memory( 
�
Ernn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2
StackPopV2Krnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/Enter^rnn/gradients/Sub*
_output_shapes

:
*
	elem_type0
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

:

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
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_accStackV2Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Const*
_output_shapes
:*
	elem_type0*7
_class-
+)loc:@rnn/rnn/while/basic_lstm_cell/Tanh_1*

stack_name 
�
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/EnterEnterBrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
_output_shapes
:*+

frame_namernn/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
�
Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2StackPushV2Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter$rnn/rnn/while/basic_lstm_cell/Tanh_1^rnn/gradients/Add*
T0*
_output_shapes

:
*
swap_memory( 
�
Grnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2
StackPopV2Mrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/Enter^rnn/gradients/Sub*
_output_shapes

:
*
	elem_type0
�
Mrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/EnterEnterBrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context*
T0
�
Grnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_depsNoOp;^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul=^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1
�
Ornn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependencyIdentity:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/MulH^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*M
_classC
A?loc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul*
_output_shapes

:

�
Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1Identity<rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1H^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*O
_classE
CAloc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1*
_output_shapes

:

�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradGrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2Ornn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradErnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
9rnn/gradients/rnn/rnn/while/Switch_2_grad_1/NextIterationNextIteration_rnn/gradients/rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
�
rnn/gradients/AddN_1AddNCrnn/gradients/rnn/rnn/while/Merge_3_grad/tuple/control_dependency_1@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Tanh_1_grad/TanhGrad*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_3_grad/b_switch*
N*
_output_shapes

:
*
T0
f
Grnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_depsNoOp^rnn/gradients/AddN_1
�
Ornn/gradients/rnn/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyIdentityrnn/gradients/AddN_1H^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
_output_shapes

:
*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_3_grad/b_switch
�
Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Identityrnn/gradients/AddN_1H^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@rnn/gradients/rnn/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:

�
8rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/MulMulOrnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyCrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2*
T0*
_output_shapes

:

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
>rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_accStackV2>rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/Const*

stack_name *
_output_shapes
:*
	elem_type0*8
_class.
,*loc:@rnn/rnn/while/basic_lstm_cell/Sigmoid
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

:
*
swap_memory( 
�
Crnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2
StackPopV2Irnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/Enter^rnn/gradients/Sub*
_output_shapes

:
*
	elem_type0
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

:

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
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_accStackV2@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Const*+
_class!
loc:@rnn/rnn/while/Identity_3*

stack_name *
_output_shapes
:*
	elem_type0
�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/EnterEnter@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*+

frame_namernn/rnn/while/while_context
�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2StackPushV2@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Enterrnn/rnn/while/Identity_3^rnn/gradients/Add*
_output_shapes

:
*
swap_memory( *
T0
�
Ernn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
StackPopV2Krnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/Enter^rnn/gradients/Sub*
	elem_type0*
_output_shapes

:

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
Mrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependencyIdentity8rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/MulF^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_deps*K
_classA
?=loc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul*
_output_shapes

:
*
T0
�
Ornn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1Identity:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1F^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_deps*M
_classC
A?loc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1*
_output_shapes

:
*
T0
�
:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/MulMulQrnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Ernn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2*
T0*
_output_shapes

:

�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@rnn/rnn/while/basic_lstm_cell/Tanh*
valueB :
���������
�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_accStackV2@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Const*
_output_shapes
:*
	elem_type0*5
_class+
)'loc:@rnn/rnn/while/basic_lstm_cell/Tanh*

stack_name 
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

:
*
swap_memory( 
�
Ernn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2
StackPopV2Krnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/Enter^rnn/gradients/Sub*
	elem_type0*
_output_shapes

:

�
Krnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/EnterEnter@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context*
T0*
is_constant(
�
<rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1MulQrnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Grnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2*
_output_shapes

:
*
T0
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
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_accStackV2Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Const*:
_class0
.,loc:@rnn/rnn/while/basic_lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:*
	elem_type0
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

:
*
swap_memory( 
�
Grnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2
StackPopV2Mrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/Enter^rnn/gradients/Sub*
	elem_type0*
_output_shapes

:

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

:

�
Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1Identity<rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1H^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1*
_output_shapes

:

�
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradCrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2Ornn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradGrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2Ornn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency*
T0*
_output_shapes

:

�
>rnn/gradients/rnn/rnn/while/basic_lstm_cell/Tanh_grad/TanhGradTanhGradErnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1*
T0*
_output_shapes

:

�
9rnn/gradients/rnn/rnn/while/Switch_3_grad_1/NextIterationNextIterationMrnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency*
_output_shapes

:
*
T0
�
:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/ShapeConst^rnn/gradients/Sub*
valueB"   
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

:
*
	keep_dims( *

Tidx0
�
<rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/ReshapeReshape8rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Sum:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Shape*
T0*
Tshape0*
_output_shapes

:

�
:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Sum_1SumDrnn/gradients/rnn/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradLrnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
>rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Reshape_1Reshape:rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Sum_1<rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
Ernn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/tuple/group_depsNoOp=^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Reshape?^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Reshape_1
�
Mrnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependencyIdentity<rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/ReshapeF^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*O
_classE
CAloc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Reshape*
_output_shapes

:

�
Ornn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependency_1Identity>rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Reshape_1F^rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/tuple/group_deps*
_output_shapes
: *
T0*Q
_classG
ECloc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/Reshape_1
�
=rnn/gradients/rnn/rnn/while/basic_lstm_cell/split_grad/concatConcatV2Frnn/gradients/rnn/rnn/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGrad>rnn/gradients/rnn/rnn/while/basic_lstm_cell/Tanh_grad/TanhGradMrnn/gradients/rnn/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependencyFrnn/gradients/rnn/rnn/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradCrnn/gradients/rnn/rnn/while/basic_lstm_cell/split_grad/concat/Const*
T0*
N*
_output_shapes

:(*

Tidx0
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

:(
�
Srnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityDrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradJ^rnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes
:(
�
>rnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMulMatMulQrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyDrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul/Enter*
_output_shapes

:*
transpose_a( *
transpose_b(*
T0
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
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Frnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Const*7
_class-
+)loc:@rnn/rnn/while/basic_lstm_cell/concat*

stack_name *
_output_shapes
:*
	elem_type0
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

:*
swap_memory( 
�
Krnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^rnn/gradients/Sub*
_output_shapes

:*
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

:
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
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterDrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
_output_shapes
:(*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context*
T0*
is_constant( *
parallel_iterations 
�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeFrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Lrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
N*
_output_shapes

:(: *
T0
�
Ernn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchFrnn/gradients/rnn/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2rnn/gradients/b_count_2* 
_output_shapes
:(:(*
T0
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
;rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/modFloorMod=rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/Const<rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
�
=rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeShapernn/rnn/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
�
>rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeNShapeNIrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2Krnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1* 
_output_shapes
::*
T0*
out_type0*
N
�
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/ConstConst*
dtype0*
_output_shapes
: *2
_class(
&$loc:@rnn/rnn/while/TensorArrayReadV3*
valueB :
���������
�
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_accStackV2Drnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const*2
_class(
&$loc:@rnn/rnn/while/TensorArrayReadV3*

stack_name *
_output_shapes
:*
	elem_type0
�
Drnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/EnterEnterDrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc*
parallel_iterations *
_output_shapes
:*+

frame_namernn/rnn/while/while_context*
T0*
is_constant(
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
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1StackV2Frnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const_1*+
_class!
loc:@rnn/rnn/while/Identity_4*

stack_name *
_output_shapes
:*
	elem_type0
�
Frnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1EnterFrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*+

frame_namernn/rnn/while/while_context
�
Lrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1StackPushV2Frnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1rnn/rnn/while/Identity_4^rnn/gradients/Add*
_output_shapes

:
*
swap_memory( *
T0
�
Krnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
StackPopV2Qrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/Enter^rnn/gradients/Sub*
	elem_type0*
_output_shapes

:

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

:

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
Rrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1Identity?rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/Slice_1I^rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/tuple/group_deps*
_output_shapes

:
*
T0*R
_classH
FDloc:@rnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/Slice_1
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
Zrnn/gradients/rnn/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn/rnn/TensorArray_1*
is_constant(*
_output_shapes
:*9

frame_name+)rnn/gradients/rnn/rnn/while/while_context*
T0*8
_class.
,*loc:@rnn/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations 
�
\rnn/gradients/rnn/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterBrnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
_output_shapes
: *9

frame_name+)rnn/gradients/rnn/rnn/while/while_context*
T0*8
_class.
,*loc:@rnn/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations 
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
Brnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Enter@rnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
parallel_iterations *
_output_shapes
: *9

frame_name+)rnn/gradients/rnn/rnn/while/while_context*
T0*
is_constant( 
�
Brnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeBrnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Hrnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
�
Arnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchBrnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2rnn/gradients/b_count_2*
_output_shapes
: : *
T0
�
>rnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddCrnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Vrnn/gradients/rnn/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
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

:

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
irnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3wrnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3 rnn/rnn/TensorArrayUnstack/rangesrnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0*4
_output_shapes"
 :������������������

�
frnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpj^rnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3C^rnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
�
nrnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityirnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3g^rnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*+
_output_shapes
:���������
*
T0*|
_classr
pnloc:@rnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
�
prnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityBrnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3g^rnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
_output_shapes
: *
T0*U
_classK
IGloc:@rnn/gradients/rnn/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
�
6rnn/gradients/rnn/rnn/transpose_grad/InvertPermutationInvertPermutationrnn/rnn/concat*
T0*
_output_shapes
:
�
.rnn/gradients/rnn/rnn/transpose_grad/transpose	Transposenrnn/gradients/rnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency6rnn/gradients/rnn/rnn/transpose_grad/InvertPermutation*
T0*+
_output_shapes
:���������
*
Tperm0
m
&rnn/gradients/rnn/Reshape_1_grad/ShapeShapernn/add*
_output_shapes
:*
T0*
out_type0
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
"rnn/gradients/rnn/add_grad/Shape_1Const*
_output_shapes
:*
valueB:
*
dtype0
�
0rnn/gradients/rnn/add_grad/BroadcastGradientArgsBroadcastGradientArgs rnn/gradients/rnn/add_grad/Shape"rnn/gradients/rnn/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
rnn/gradients/rnn/add_grad/SumSum(rnn/gradients/rnn/Reshape_1_grad/Reshape0rnn/gradients/rnn/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
$rnn/gradients/rnn/add_grad/Reshape_1Reshape rnn/gradients/rnn/add_grad/Sum_1"rnn/gradients/rnn/add_grad/Shape_1*
Tshape0*
_output_shapes
:
*
T0
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
$rnn/gradients/rnn/MatMul_grad/MatMulMatMul3rnn/gradients/rnn/add_grad/tuple/control_dependencylayer/weights/Variable/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
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
6rnn/gradients/rnn/MatMul_grad/tuple/control_dependencyIdentity$rnn/gradients/rnn/MatMul_grad/MatMul/^rnn/gradients/rnn/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*7
_class-
+)loc:@rnn/gradients/rnn/MatMul_grad/MatMul
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
rnn/beta1_power/AssignAssignrnn/beta1_powerrnn/beta1_power/initial_value*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(
|
rnn/beta1_power/readIdentityrnn/beta1_power*
_output_shapes
: *
T0*(
_class
loc:@layer/biases/Variable
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
VariableV2*
_output_shapes
: *
shared_name *(
_class
loc:@layer/biases/Variable*
	container *
shape: *
dtype0
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
&rnn/layer/weights/Variable/Adam/AssignAssignrnn/layer/weights/Variable/Adam1rnn/layer/weights/Variable/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable
�
$rnn/layer/weights/Variable/Adam/readIdentityrnn/layer/weights/Variable/Adam*)
_class
loc:@layer/weights/Variable*
_output_shapes

:
*
T0
�
3rnn/layer/weights/Variable/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:
*)
_class
loc:@layer/weights/Variable*
valueB
*    
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
(rnn/layer/weights/Variable/Adam_1/AssignAssign!rnn/layer/weights/Variable/Adam_13rnn/layer/weights/Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

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
VariableV2*+
_class!
loc:@layer/weights/Variable_1*
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
(rnn/layer/weights/Variable_1/Adam/AssignAssign!rnn/layer/weights/Variable_1/Adam3rnn/layer/weights/Variable_1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1
�
&rnn/layer/weights/Variable_1/Adam/readIdentity!rnn/layer/weights/Variable_1/Adam*+
_class!
loc:@layer/weights/Variable_1*
_output_shapes

:
*
T0
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
*rnn/layer/weights/Variable_1/Adam_1/AssignAssign#rnn/layer/weights/Variable_1/Adam_15rnn/layer/weights/Variable_1/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:

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
VariableV2*
shape:
*
dtype0*
_output_shapes
:
*
shared_name *(
_class
loc:@layer/biases/Variable*
	container 
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
'rnn/layer/biases/Variable/Adam_1/AssignAssign rnn/layer/biases/Variable/Adam_12rnn/layer/biases/Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

�
%rnn/layer/biases/Variable/Adam_1/readIdentity rnn/layer/biases/Variable/Adam_1*(
_class
loc:@layer/biases/Variable*
_output_shapes
:
*
T0
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
'rnn/layer/biases/Variable_1/Adam/AssignAssign rnn/layer/biases/Variable_1/Adam2rnn/layer/biases/Variable_1/Adam/Initializer/zeros*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
%rnn/layer/biases/Variable_1/Adam/readIdentity rnn/layer/biases/Variable_1/Adam*
T0**
_class 
loc:@layer/biases/Variable_1*
_output_shapes
:
�
4rnn/layer/biases/Variable_1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:**
_class 
loc:@layer/biases/Variable_1*
valueB*    
�
"rnn/layer/biases/Variable_1/Adam_1
VariableV2**
_class 
loc:@layer/biases/Variable_1*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
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
9rnn/rnn/rnn/basic_lstm_cell/kernel/Adam/Initializer/zerosConst*
_output_shapes

:(*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
valueB(*    *
dtype0
�
'rnn/rnn/rnn/basic_lstm_cell/kernel/Adam
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
VariableV2*
shared_name *1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
	container *
shape
:(*
dtype0*
_output_shapes

:(
�
0rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1/AssignAssign)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1;rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:(*
use_locking(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel
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
VariableV2*
shared_name */
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
	container *
shape:(*
dtype0*
_output_shapes
:(
�
,rnn/rnn/rnn/basic_lstm_cell/bias/Adam/AssignAssign%rnn/rnn/rnn/basic_lstm_cell/bias/Adam7rnn/rnn/rnn/basic_lstm_cell/bias/Adam/Initializer/zeros*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
*rnn/rnn/rnn/basic_lstm_cell/bias/Adam/readIdentity%rnn/rnn/rnn/basic_lstm_cell/bias/Adam*
_output_shapes
:(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias
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
VariableV2*
shape:(*
dtype0*
_output_shapes
:(*
shared_name */
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
	container 
�
.rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1/AssignAssign'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_19rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:(*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias
�
,rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1/readIdentity'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1*
_output_shapes
:(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias
[
rnn/Adam/learning_rateConst*
valueB
 *��8*
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
rnn/Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
U
rnn/Adam/epsilonConst*
_output_shapes
: *
valueB
 *w�+2*
dtype0
�
0rnn/Adam/update_layer/weights/Variable/ApplyAdam	ApplyAdamlayer/weights/Variablernn/layer/weights/Variable/Adam!rnn/layer/weights/Variable/Adam_1rnn/beta1_power/readrnn/beta2_power/readrnn/Adam/learning_raternn/Adam/beta1rnn/Adam/beta2rnn/Adam/epsilon8rnn/gradients/rnn/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@layer/weights/Variable*
use_nesterov( *
_output_shapes

:

�
2rnn/Adam/update_layer/weights/Variable_1/ApplyAdam	ApplyAdamlayer/weights/Variable_1!rnn/layer/weights/Variable_1/Adam#rnn/layer/weights/Variable_1/Adam_1rnn/beta1_power/readrnn/beta2_power/readrnn/Adam/learning_raternn/Adam/beta1rnn/Adam/beta2rnn/Adam/epsilon:rnn/gradients/rnn/MatMul_1_grad/tuple/control_dependency_1*
_output_shapes

:
*
use_locking( *
T0*+
_class!
loc:@layer/weights/Variable_1*
use_nesterov( 
�
/rnn/Adam/update_layer/biases/Variable/ApplyAdam	ApplyAdamlayer/biases/Variablernn/layer/biases/Variable/Adam rnn/layer/biases/Variable/Adam_1rnn/beta1_power/readrnn/beta2_power/readrnn/Adam/learning_raternn/Adam/beta1rnn/Adam/beta2rnn/Adam/epsilon5rnn/gradients/rnn/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:
*
use_locking( *
T0*(
_class
loc:@layer/biases/Variable
�
1rnn/Adam/update_layer/biases/Variable_1/ApplyAdam	ApplyAdamlayer/biases/Variable_1 rnn/layer/biases/Variable_1/Adam"rnn/layer/biases/Variable_1/Adam_1rnn/beta1_power/readrnn/beta2_power/readrnn/Adam/learning_raternn/Adam/beta1rnn/Adam/beta2rnn/Adam/epsilon7rnn/gradients/rnn/preds_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@layer/biases/Variable_1*
use_nesterov( *
_output_shapes
:
�
8rnn/Adam/update_rnn/rnn/basic_lstm_cell/kernel/ApplyAdam	ApplyAdamrnn/rnn/basic_lstm_cell/kernel'rnn/rnn/rnn/basic_lstm_cell/kernel/Adam)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn/beta1_power/readrnn/beta2_power/readrnn/Adam/learning_raternn/Adam/beta1rnn/Adam/beta2rnn/Adam/epsilonErnn/gradients/rnn/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
use_nesterov( *
_output_shapes

:(*
use_locking( 
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
rnn/save/RestoreV2	RestoreV2rnn/save/Constrnn/save/RestoreV2/tensor_names#rnn/save/RestoreV2/shape_and_slices"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
�
rnn/save/AssignAssignlayer/biases/Variablernn/save/RestoreV2*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

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
rnn/save/Assign_2Assignlayer/weights/Variablernn/save/RestoreV2:2*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:
*
use_locking(
�
rnn/save/Assign_3Assignlayer/weights/Variable_1rnn/save/RestoreV2:3*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:

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
rnn/save/Assign_5Assignrnn/beta2_powerrnn/save/RestoreV2:5*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
: 
�
rnn/save/Assign_6Assignrnn/layer/biases/Variable/Adamrnn/save/RestoreV2:6*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

�
rnn/save/Assign_7Assign rnn/layer/biases/Variable/Adam_1rnn/save/RestoreV2:7*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

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
rnn/save/Assign_9Assign"rnn/layer/biases/Variable_1/Adam_1rnn/save/RestoreV2:9*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:
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
rnn/save/Assign_12Assign!rnn/layer/weights/Variable_1/Adamrnn/save/RestoreV2:12*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:

�
rnn/save/Assign_13Assign#rnn/layer/weights/Variable_1/Adam_1rnn/save/RestoreV2:13*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1
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
rnn/save/Assign_15Assignrnn/rnn/basic_lstm_cell/kernelrnn/save/RestoreV2:15*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(*
T0
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
rnn/save/Assign_17Assign'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1rnn/save/RestoreV2:17*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
rnn/save/Assign_18Assign'rnn/rnn/rnn/basic_lstm_cell/kernel/Adamrnn/save/RestoreV2:18*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(*
T0
�
rnn/save/Assign_19Assign)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn/save/RestoreV2:19*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(*
T0
�
rnn/save/restore_allNoOp^rnn/save/Assign^rnn/save/Assign_1^rnn/save/Assign_10^rnn/save/Assign_11^rnn/save/Assign_12^rnn/save/Assign_13^rnn/save/Assign_14^rnn/save/Assign_15^rnn/save/Assign_16^rnn/save/Assign_17^rnn/save/Assign_18^rnn/save/Assign_19^rnn/save/Assign_2^rnn/save/Assign_3^rnn/save/Assign_4^rnn/save/Assign_5^rnn/save/Assign_6^rnn/save/Assign_7^rnn/save/Assign_8^rnn/save/Assign_9
M
rnn/probPlaceholder*
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
value3B1 B+_temp_c2511c204a964ba5bd2b054141cee3ec/part*
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
rnn/save_1/control_dependencyIdentityrnn/save_1/ShardedFilename^rnn/save_1/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*-
_class#
!loc:@rnn/save_1/ShardedFilename
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
rnn/save_1/Assign_1Assignlayer/biases/Variable_1rnn/save_1/RestoreV2:1*
validate_shape(*
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@layer/biases/Variable_1
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
rnn/save_1/Assign_3Assignlayer/weights/Variable_1rnn/save_1/RestoreV2:3*
use_locking(*
T0*+
_class!
loc:@layer/weights/Variable_1*
validate_shape(*
_output_shapes

:

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
rnn/save_1/Assign_6Assignrnn/layer/biases/Variable/Adamrnn/save_1/RestoreV2:6*
use_locking(*
T0*(
_class
loc:@layer/biases/Variable*
validate_shape(*
_output_shapes
:

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
rnn/save_1/Assign_8Assign rnn/layer/biases/Variable_1/Adamrnn/save_1/RestoreV2:8**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
rnn/save_1/Assign_9Assign"rnn/layer/biases/Variable_1/Adam_1rnn/save_1/RestoreV2:9*
T0**
_class 
loc:@layer/biases/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
rnn/save_1/Assign_10Assignrnn/layer/weights/Variable/Adamrnn/save_1/RestoreV2:10*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

�
rnn/save_1/Assign_11Assign!rnn/layer/weights/Variable/Adam_1rnn/save_1/RestoreV2:11*
use_locking(*
T0*)
_class
loc:@layer/weights/Variable*
validate_shape(*
_output_shapes

:

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
rnn/save_1/Assign_14Assignrnn/rnn/basic_lstm_cell/biasrnn/save_1/RestoreV2:14*
use_locking(*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(
�
rnn/save_1/Assign_15Assignrnn/rnn/basic_lstm_cell/kernelrnn/save_1/RestoreV2:15*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(*
use_locking(
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
rnn/save_1/Assign_17Assign'rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1rnn/save_1/RestoreV2:17*
T0*/
_class%
#!loc:@rnn/rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:(*
use_locking(
�
rnn/save_1/Assign_18Assign'rnn/rnn/rnn/basic_lstm_cell/kernel/Adamrnn/save_1/RestoreV2:18*
use_locking(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
rnn/save_1/Assign_19Assign)rnn/rnn/rnn/basic_lstm_cell/kernel/Adam_1rnn/save_1/RestoreV2:19*
use_locking(*
T0*1
_class'
%#loc:@rnn/rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:(
�
rnn/save_1/restore_shardNoOp^rnn/save_1/Assign^rnn/save_1/Assign_1^rnn/save_1/Assign_10^rnn/save_1/Assign_11^rnn/save_1/Assign_12^rnn/save_1/Assign_13^rnn/save_1/Assign_14^rnn/save_1/Assign_15^rnn/save_1/Assign_16^rnn/save_1/Assign_17^rnn/save_1/Assign_18^rnn/save_1/Assign_19^rnn/save_1/Assign_2^rnn/save_1/Assign_3^rnn/save_1/Assign_4^rnn/save_1/Assign_5^rnn/save_1/Assign_6^rnn/save_1/Assign_7^rnn/save_1/Assign_8^rnn/save_1/Assign_9
9
rnn/save_1/restore_allNoOp^rnn/save_1/restore_shard"N
rnn/save_1/Const:0rnn/save_1/Identity:0rnn/save_1/restore_all (5 @F8"
train_op


rnn/Adam"�
	variables��
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
)rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1:0.rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1/Assign.rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1/read:02;rnn/rnn/rnn/basic_lstm_cell/bias/Adam_1/Initializer/zeros:0"�<
while_context�<�<
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
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc:0Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Enter:0B
rnn/rnn/TensorArray_1:0'rnn/rnn/while/TensorArrayReadV3/Enter:07
rnn/rnn/strided_slice_1:0rnn/rnn/while/Less/Enter:0�
@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc:0@rnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_grad/Mul/Enter:0�
Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc:0Brnn/gradients/rnn/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Enter:0�
Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1:0Hrnn/gradients/rnn/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1:0U
%rnn/rnn/basic_lstm_cell/kernel/read:0,rnn/rnn/while/basic_lstm_cell/MatMul/Enter:0q
Drnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0)rnn/rnn/while/TensorArrayReadV3/Enter_1:0Rrnn/rnn/while/Enter:0Rrnn/rnn/while/Enter_1:0Rrnn/rnn/while/Enter_2:0Rrnn/rnn/while/Enter_3:0Rrnn/rnn/while/Enter_4:0Rrnn/gradients/f_count_1:0Zrnn/rnn/strided_slice_1:0"�
trainable_variables��
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