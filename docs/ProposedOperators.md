# Proposed General-Purpose Operators for Data Processing Pipelines

The following document contains an outline of the operators that are proposed for defining general-purpose
data processing pipelines. Some operators may overlap in definition with existing operators, but are included
to indicate the full set of types that should be covered. Some may also have slightly narrower definitions than
some existing, overlapping, operators.

## Scalar Types

### Numeric Types

All standard arithmetic operators.
All standard relational and equality operators.

### Boolean

    not(x: boolean) : boolean
    and(x: boolean, y: boolean) : boolean
    or(x:  boolean, y: boolean) : boolean

## Nullables

Testing whether a value is null or not.

    isNull(x: nullable<T>) : boolean

Retrieving the value inside a nullable, providing a default for the case when it's null.

    getValue(x: nullable<T>, default: T) : T

Creating a nullable value, and collections of nullable values.

    nullable<T>()) : nullable<T>
    nullable(value : T) : nullable<T>
    nullableSequence(value : sequence<T>) : sequence<nullable<T>>
    nullableMap(value : map<S,T>) : map<S,nullable<T>>

### Impute 

impute will replace all null values with a value of the underlying type. For numeric values, a null value may
be provided. For non-tensor collections of any type, nullable<T> is supported.
The output collection has the same shape as the input.

    impute(input: tensor<T>, default: T, null: T) : tensor<T>
    impute(input: sequence<T>, default: T, null: T) : sequence<T>
    impute(input: map<S,T>, default: T, null: T) : map<S,T>
    impute(input: sequence<nullable<T>>, default: T) : sequence<T>
    impute(input: map<S,nullable<T>>,    default: T) : map<S,U>

For floating point numbers, NaN and Inf are supported as special null values to be imputed. Those values cannot
be handled using the generic inpute() operator.

    imputeNaN(input: tensor<T>, default: T) : sequence<T>
    imputeNaN(input: sequence<T>, default: T) : sequence<T>
    imputeNaN(input: map<S,T>,    default: T) : map<S,U>
    imputeInf(input: tensor<T>, default: T) : sequence<T>
    imputeInf(input: sequence<T>, default: T) : sequence<T>
    imputeInf(input: map<S,T>,    default: T) : map<S,U>

## Records and Tuples

Records and tuples are constructed from a variadic input list, denoting values of types T<sub>0</sub>, T<sub>1</sub>,...,T<sub>n</sub>.
For records, the record name and the field names are determined by the type string attribute. The values must be provided in the same order that the corresponding fiels are
listed in the type string; their respective types must exactly match the corresponding types in the type string attribute.

    tuple(type: string, values: ...) : tuple<T0,T1,...,Tn>
    record(type: string, values: ...) : record<{name},{fieldNames},{T0,T1,...,Tn}>

Accessing fields of tuples or records results in a type that depends on the field that is accessed. All field indices and names are static, so
the type can be determined through ONNX type inference.

    get(s: tuple<T0,T1,...,Tn>, i: int) : T[i]
    get(s: record<{name},{fieldNames},{T0,T1,...,Tn}>, field: string) : T[{fieldName}]

Record and tuple fields are modified in an immutable fashion, creating a new value with only some fields modified. Note the existence
of operators to modify multiple fields in one fell swoop.

    set(s: tuple<T0,T1,...,Tn>, i: int, value: T[i]) : tuple<T0,T1,...,Tn>
    set(s: tuple<T0,T1,...,Tn>, i: sequence<int>, values: ...) : tuple<T0,T1,...,Tn>

    set(s: record<{name},{fieldNames},{T0,T1,...,Tn}>, field: string, value: T[{fieldName}]) : record<{name},{fieldNames},{T0,T1,...,Tn}>
    set(s: record<{name},{fieldNames},{T0,T1,...,Tn}>, field: sequence<string>, values: ...) : record<{name},{fieldNames},{T0,T1,...,Tn}>

## Collections

### Fold

fold will accumulate a value, starting with the 'basis' argument, and then apply the function
to the value and each element in the collection. The accumulate value is the result of the 
operator. S and T are usually the same type, but do not have to be.

    fold(input: tensor<T>,   basis: S, tfrm: (S,T)->S) : S
    fold(input: sequence<T>, basis: S, tfrm: (S,T)->S) : S

### Map

map will produce a collection from a source collection and a function. 
The output collection has the same shape as the input.

    map(input: tensor<T>,   tfrm: T->U) : tensor<U>
    map(input: sequence<T>, tfrm: T->U) : sequence<U>
    map(input: map<S,T>,    tfrm: T->U) : map<S,U>

### Scale

scale will produce a collection from a source collection, a scale factor, and an offset.
The output collection has the same shape as the input.

    scale(input: tensor<T>,   scale: T, offset: T) : tensor<T>
    scale(input: sequence<T>, scale: T, offset: T) : sequence<T>
    scale(input: map<S,T>,    scale: T, offset: T) : map<S,T>

### ZipMap

zipmap will produce a collection from two source collections and a function. 
The input shapes will be unified by the usual broadcasting semantics.
The output collection has the same shape as the inputs.

    zipmap(x: tensor<S>,   y: tensor<T>,   tfrm: (S,T)->U) : tensor<U>
    zipmap(x: sequence<S>, y: sequence<T>, tfrm: (S,T)->U) : sequence<U>
    zipmap(x: map<R,S>,    y: map<R,T>,    tfrm: (S,T)->U) : map<S,U>


## Tensors

### Dimensions

Get the total number of elements, or the size of each dimension for a given tensor.

    len(t: tensor<T>) : int
    dims(t: tensor<T>) : sequence<int>

### Transpose & Permute

transpose will flip the row / column orientation of 2-dimensional tensors.

    transpose(input: tensor<T;[N,M]>) : tensor<T;[M,N]> 

permute is matrix transposition generalized to N dimensions. For matrices,
it is equivalent to transpose()

    permute(input: tensor<T>, permutation: sequence<int>) : tensor<T>

## Sequences

### Basics

    len  (s: sequence<T>) : int

### Value Retrieval

    get  (s: sequence<T>, i: int) : T

    slice(s: sequence<T>, start: int, len: int=-1) : sequence<T>
    remove(s: sequence<T>, start: int, len: int=-1) : sequence<T>
    filter(s: sequence<T>, predicate: T->boolean) : sequence<T>
    window(s: sequence<T>, size: int, step: int) : sequence<sequence<T>>

### Adding Values

    append(s: sequence<T>, elem: T) : sequence<T>
    insert(s: sequence<T>, elem: T, at: int) : sequence<T>
    replace(s: sequence<T>, elem: T, at: int) : sequence<T>

### Concatenation

    concat(x: sequence<T>, y: sequence<T>) : sequence<T>
    flatten(x: sequence<sequence<T>>) : sequence<T>

### Others

The comparer function should return -1,0,1 in the usual fashion.

    sort(s: sequence<T>, comparer: (T,T)->int) : sequence<T> 

    min(s: sequence<T>, comparer: (T,T)->int) : T
    max(s: sequence<T>, comparer: (T,T)->int) : T
    argmax(s: sequence<T>, comparer: (T,T)->int) : int
    argmin(s: sequence<T>, comparer: (T,T)->int) : int

    any(s: sequence<T>, predicate: T->boolean) : boolean
    all(s: sequence<T>, predicate: T->boolean) : boolean

## Maps

TBD

## Dates and Time

TBD

## Strings

#### Basics

    len(s: string) : int

### Conversions

Operators to convert to and from strings.

    toString(i: int) : string
    toString(b: boolean) : string
    toString(f: float) : string
    toString(d: double) : string

    parseInt(s: string) : int
    parseBoolean(s: string) : boolean
    parseFloat(s: string) : float
    parseDouble(s: string) : double

### Substrings

    substring(s: string, start: int, length: int) : string

    contains(s: string, substring: string, caseSense: boolean) : boolean
    startswith(s: string, substring: string, caseSense: boolean) : boolean
    endswith(s: string, substring: string, caseSense: boolean) : boolean

    count(s: string, substring: string, caseSense: boolean) : int
    index(s: string, substring: string, caseSense: boolean) : int
    rindex(s: string, substring: string, caseSense: boolean) : int

### Strings and String Arrays

    join(strings: sequence<string>, separator: string) : string
    split(s: string, sep: string) : sequence<string>

### Content Manipulation

These operators will transform the value of a string.

    toLower(s: string) : string
    toUpper(s: string) : string

    strip (s: string, chars: string) : string
    lstrip(s: string, chars: string) : string
    rstrip(s: string, chars: string) : string

## Regular Expressions

TBD -- not known whether we need them.