/// set matriz=##class(Matriz).%New()
Class Matriz Extends %Persistent [ ClassType = persistent, ProcedureBlock ]
{

Property linhas As %Integer [ Required ];

Property colunas As %Integer [ Required ];

Property matriz As %Library.DynamicArray [ Required ];

Method %OnNew(linhas As %Integer = 1, colunas As %Integer = 1, startValue, falgRandom) As %Status
{
	set ..linhas=linhas
	set ..colunas=colunas
	set startValue=$get(startValue)
	if startValue="" set startValue=0
	set falgRandom=$get(falgRandom)
	
	set novaMatriz=[]
	for m=0:1:..linhas-1 {
		set novoArray=[]
		for l=0:1:..colunas-1 {
			if falgRandom set startValue=$random(1000)/1000
			do novoArray.%Push(startValue)
		}
		do novaMatriz.%Push(novoArray)
	}
	set ..matriz=novaMatriz
	;
	Quit $$$OK
}

// set sc=##class(Matriz).Teste()

ClassMethod Teste() As %Status
{
	set sc=$$$OK
	set matriz=##class(Matriz).%New(5,2)
	
	w !,matriz.matriz
	w !,matriz.matriz.%ToJSON()
	set funcao="MapDobrar"
	set matrizMap=matriz.Map(..%ClassName(1),funcao)
	w !,matrizMap.matriz
	w !,matrizMap.matriz.%ToJSON()
	
	set matrizMap2=matrizMap.Map(..%ClassName(1),funcao)
	w !,matrizMap2.matriz
	w !,matrizMap2.matriz.%ToJSON()
	
	set matrizMap3=##class(Matriz).Multiplicar(matrizMap, matrizMap2)
	w !,matrizMap3.matriz
	w !,matrizMap3.matriz.%ToJSON()
	
	quit sc
}

// set matriz=##class(Matriz).ArrayToMatriz([7,7])

ClassMethod ArrayToMatriz(array As %Library.DynamicArray) As Matriz
{
	set tamanhoArray=array.%Size()
	set matriz=##class(Matriz).%New(tamanhoArray,1)
	
	for m=0:1:tamanhoArray-1 {
		set valorArray=[].%Push(array.%Get(m))
		do matriz.matriz.%Set(m,valorArray)
	}
	quit matriz
}

// set matriz=##class(Matriz).MatrizToArray([7,7])

ClassMethod MatrizToArray(matrizA As Matriz) As %Library.DynamicArray
{
	set novoArray=[]
	for m=0:1:matrizA.linhas-1 {
		
		for l=0:1:matrizA.colunas-1 {
			
			do novoArray.%Push(matrizA.matriz.%Get(m).%Get(l))
			
		}
		
	}
	quit novoArray
}

// set matriz=##class(Matriz).Adicionar(matrizA, matrizB)

ClassMethod Adicionar(matrizA As Matriz, matrizB As Matriz) As Matriz
{
	set novaMatriz=##class(Matriz).%New(matrizA.linhas,matrizA.colunas)
	
	set novaMatriz=novaMatriz.Map(..%ClassName(1),"MapSomarMatriz", matrizA, matrizB)
	quit novaMatriz
}

// set matriz=##class(Matriz).Subtrair(matrizA, matrizB)

ClassMethod Subtrair(matrizA As Matriz, matrizB As Matriz) As Matriz
{
	set novaMatriz=##class(Matriz).%New(matrizA.linhas,matrizA.colunas)
	
	set novaMatriz=novaMatriz.Map(..%ClassName(1),"MapSubtrairMatriz", matrizA, matrizB)
	quit novaMatriz
}

// set matriz=##class(Matriz).Multiplicar(matrizA, matrizB)

ClassMethod Multiplicar(matrizA As Matriz, matrizB As Matriz) As Matriz
{
	set novaMatriz=##class(Matriz).%New(matrizA.linhas,matrizB.colunas)
	
	set novaMatriz=novaMatriz.Map(..%ClassName(1),"MapMultiplicarMatriz", matrizA, matrizB)
	quit novaMatriz
}

// set matriz=##class(Matriz).Transpor(matrizA)

ClassMethod Transpor(matrizA As Matriz) As Matriz
{
	set novaMatriz=##class(Matriz).%New(matrizA.colunas, matrizA.linhas)
	
	set novaMatriz=novaMatriz.Map(..%ClassName(1),"MapTranspor", matrizA)
	quit novaMatriz
}

// set matriz=##class(Matriz).Hadamard(matrizA, matrizB)

ClassMethod Hadamard(matrizA As Matriz, matrizB As Matriz) As Matriz
{
	set novaMatriz=##class(Matriz).%New(matrizA.linhas, matrizB.colunas)
	
	set novaMatriz=novaMatriz.Map(..%ClassName(1),"MapHadamard", matrizA, matrizB)
	quit novaMatriz
}

// set matriz=##class(Matriz).EscalarMultiplicar(matrizA, valorEscalar)

ClassMethod EscalarMultiplicar(matrizA As Matriz, valorEscalar As %Float) As Matriz
{
	set novaMatriz=##class(Matriz).%New(matrizA.linhas, matrizA.colunas)
	set matrizB=##class(Matriz).%New(matrizA.linhas, matrizA.colunas, valorEscalar)
	
	set novaMatriz=novaMatriz.Map(..%ClassName(1),"MapMultiplicarMatriz", matrizA, matrizB)
	quit novaMatriz
}

// set matrizMap=matriz.Map(classe,funcao)

Method Map(classe, funcao As %String, matrizA As Matriz, matrizB As Matriz) As Matriz
{
	set matrizA=$get(matrizA)
	set matrizB=$get(matrizB)
	set novaMatriz=##class(Matriz).%New(..linhas,..colunas)
	
	for m=0:1:novaMatriz.linhas-1 {
		set arrayColunas=..matriz.%Get(m)
		
		for l=0:1:novaMatriz.colunas-1 {
			set valorLinhaColuna=arrayColunas.%Get(l)
			
			set novoValor=$classmethod(classe, funcao, valorLinhaColuna, m, l, matrizA, matrizB)
			
			do arrayColunas.%Set(l,novoValor)
			
		}
		
		do novaMatriz.matriz.%Set(m,arrayColunas)
		
	}
	quit novaMatriz
}

ClassMethod MapDobrar(valorLinha As %Float, m As %Integer, l As %Integer, matrizA As Matriz, matrizB As Matriz) As %Float
{
	quit valorLinha*2
}

ClassMethod MapHadamard(valorLinha As %Float, m As %Integer, l As %Integer, matrizA As Matriz, matrizB As Matriz) As %Float
{
	quit matrizA.matriz.%Get(m).%Get(l)*matrizB.matriz.%Get(m).%Get(l)
}

ClassMethod MapTranspor(valorLinha As %Float, m As %Integer, l As %Integer, matrizA As Matriz, matrizB As Matriz) As %Float
{
	quit matrizA.matriz.%Get(l).%Get(m)
}

ClassMethod MapMultiplicarMatriz(valorLinha As %Float, m As %Integer, l As %Integer, matrizA As Matriz, matrizB As Matriz) As %Float
{
	set soma = 0
	for k=0:1:matrizA.colunas-1 {
		set valorMatriz1 = matrizA.matriz.%Get(m).%Get(k)
		set valorMatriz2 = matrizB.matriz.%Get(k).%Get(l)
		set soma=soma+(valorMatriz1 * valorMatriz2)
	}
	quit soma
}

ClassMethod MapSomarMatriz(valorLinha As %Float, m As %Integer, l As %Integer, matrizA As Matriz, matrizB As Matriz) As %Float
{
	quit matrizA.matriz.%Get(m).%Get(l)+matrizB.matriz.%Get(m).%Get(l)
}

ClassMethod MapSubtrairMatriz(valorLinha As %Float, m As %Integer, l As %Integer, matrizA As Matriz, matrizB As Matriz) As %Float
{
	quit matrizA.matriz.%Get(m).%Get(l)-matrizB.matriz.%Get(m).%Get(l)
}

Storage Default
{
<Data name="MatrizDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>linhas</Value>
</Value>
<Value name="3">
<Value>colunas</Value>
</Value>
<Value name="4">
<Value>matriz</Value>
</Value>
</Data>
<DataLocation>^MatrizD</DataLocation>
<DefaultData>MatrizDefaultData</DefaultData>
<IdLocation>^MatrizD</IdLocation>
<IndexLocation>^MatrizI</IndexLocation>
<StreamLocation>^MatrizS</StreamLocation>
<Type>%Library.CacheStorage</Type>
}

}
/// set matriz=##class(Matriz).%New()
Class Matriz Extends %Persistent [ ClassType = persistent, ProcedureBlock ]
{

Property linhas As %Integer [ Required ];

Property colunas As %Integer [ Required ];

Property matriz As %Library.DynamicArray [ Required ];

Method %OnNew(linhas As %Integer = 1, colunas As %Integer = 1, startValue, falgRandom) As %Status
{
	set ..linhas=linhas
	set ..colunas=colunas
	set startValue=$get(startValue)
	if startValue="" set startValue=0
	set falgRandom=$get(falgRandom)
	
	set novaMatriz=[]
	for m=0:1:..linhas-1 {
		set novoArray=[]
		for l=0:1:..colunas-1 {
			if falgRandom set startValue=$random(1000)/1000
			do novoArray.%Push(startValue)
		}
		do novaMatriz.%Push(novoArray)
	}
	set ..matriz=novaMatriz
	;
	Quit $$$OK
}

// set sc=##class(Matriz).Teste()

ClassMethod Teste() As %Status
{
	set sc=$$$OK
	set matriz=##class(Matriz).%New(5,2)
	
	w !,matriz.matriz
	w !,matriz.matriz.%ToJSON()
	set funcao="MapDobrar"
	set matrizMap=matriz.Map(..%ClassName(1),funcao)
	w !,matrizMap.matriz
	w !,matrizMap.matriz.%ToJSON()
	
	set matrizMap2=matrizMap.Map(..%ClassName(1),funcao)
	w !,matrizMap2.matriz
	w !,matrizMap2.matriz.%ToJSON()
	
	set matrizMap3=##class(Matriz).Multiplicar(matrizMap, matrizMap2)
	w !,matrizMap3.matriz
	w !,matrizMap3.matriz.%ToJSON()
	
	quit sc
}

// set matriz=##class(Matriz).ArrayToMatriz([7,7])

ClassMethod ArrayToMatriz(array As %Library.DynamicArray) As Matriz
{
	set tamanhoArray=array.%Size()
	set matriz=##class(Matriz).%New(tamanhoArray,1)
	
	for m=0:1:tamanhoArray-1 {
		set valorArray=[].%Push(array.%Get(m))
		do matriz.matriz.%Set(m,valorArray)
	}
	quit matriz
}

// set matriz=##class(Matriz).MatrizToArray([7,7])

ClassMethod MatrizToArray(matrizA As Matriz) As %Library.DynamicArray
{
	set novoArray=[]
	for m=0:1:matrizA.linhas-1 {
		
		for l=0:1:matrizA.colunas-1 {
			
			do novoArray.%Push(matrizA.matriz.%Get(m).%Get(l))
			
		}
		
	}
	quit novoArray
}

// set matriz=##class(Matriz).Adicionar(matrizA, matrizB)

ClassMethod Adicionar(matrizA As Matriz, matrizB As Matriz) As Matriz
{
	set novaMatriz=##class(Matriz).%New(matrizA.linhas,matrizA.colunas)
	
	set novaMatriz=novaMatriz.Map(..%ClassName(1),"MapSomarMatriz", matrizA, matrizB)
	quit novaMatriz
}

// set matriz=##class(Matriz).Subtrair(matrizA, matrizB)

ClassMethod Subtrair(matrizA As Matriz, matrizB As Matriz) As Matriz
{
	set novaMatriz=##class(Matriz).%New(matrizA.linhas,matrizA.colunas)
	
	set novaMatriz=novaMatriz.Map(..%ClassName(1),"MapSubtrairMatriz", matrizA, matrizB)
	quit novaMatriz
}

// set matriz=##class(Matriz).Multiplicar(matrizA, matrizB)

ClassMethod Multiplicar(matrizA As Matriz, matrizB As Matriz) As Matriz
{
	set novaMatriz=##class(Matriz).%New(matrizA.linhas,matrizB.colunas)
	
	set novaMatriz=novaMatriz.Map(..%ClassName(1),"MapMultiplicarMatriz", matrizA, matrizB)
	quit novaMatriz
}

// set matriz=##class(Matriz).Transpor(matrizA)

ClassMethod Transpor(matrizA As Matriz) As Matriz
{
	set novaMatriz=##class(Matriz).%New(matrizA.colunas, matrizA.linhas)
	
	set novaMatriz=novaMatriz.Map(..%ClassName(1),"MapTranspor", matrizA)
	quit novaMatriz
}

// set matriz=##class(Matriz).Hadamard(matrizA, matrizB)

ClassMethod Hadamard(matrizA As Matriz, matrizB As Matriz) As Matriz
{
	set novaMatriz=##class(Matriz).%New(matrizA.linhas, matrizB.colunas)
	
	set novaMatriz=novaMatriz.Map(..%ClassName(1),"MapHadamard", matrizA, matrizB)
	quit novaMatriz
}

// set matriz=##class(Matriz).EscalarMultiplicar(matrizA, valorEscalar)

ClassMethod EscalarMultiplicar(matrizA As Matriz, valorEscalar As %Float) As Matriz
{
	set novaMatriz=##class(Matriz).%New(matrizA.linhas, matrizA.colunas)
	set matrizB=##class(Matriz).%New(matrizA.linhas, matrizA.colunas, valorEscalar)
	
	set novaMatriz=novaMatriz.Map(..%ClassName(1),"MapMultiplicarMatriz", matrizA, matrizB)
	quit novaMatriz
}

// set matrizMap=matriz.Map(classe,funcao)

Method Map(classe, funcao As %String, matrizA As Matriz, matrizB As Matriz) As Matriz
{
	set matrizA=$get(matrizA)
	set matrizB=$get(matrizB)
	set novaMatriz=##class(Matriz).%New(..linhas,..colunas)
	
	for m=0:1:novaMatriz.linhas-1 {
		set arrayColunas=..matriz.%Get(m)
		
		for l=0:1:novaMatriz.colunas-1 {
			set valorLinhaColuna=arrayColunas.%Get(l)
			
			set novoValor=$classmethod(classe, funcao, valorLinhaColuna, m, l, matrizA, matrizB)
			
			do arrayColunas.%Set(l,novoValor)
			
		}
		
		do novaMatriz.matriz.%Set(m,arrayColunas)
		
	}
	quit novaMatriz
}

ClassMethod MapDobrar(valorLinha As %Float, m As %Integer, l As %Integer, matrizA As Matriz, matrizB As Matriz) As %Float
{
	quit valorLinha*2
}

ClassMethod MapHadamard(valorLinha As %Float, m As %Integer, l As %Integer, matrizA As Matriz, matrizB As Matriz) As %Float
{
	quit matrizA.matriz.%Get(m).%Get(l)*matrizB.matriz.%Get(m).%Get(l)
}

ClassMethod MapTranspor(valorLinha As %Float, m As %Integer, l As %Integer, matrizA As Matriz, matrizB As Matriz) As %Float
{
	quit matrizA.matriz.%Get(l).%Get(m)
}

ClassMethod MapMultiplicarMatriz(valorLinha As %Float, m As %Integer, l As %Integer, matrizA As Matriz, matrizB As Matriz) As %Float
{
	set soma = 0
	for k=0:1:matrizA.colunas-1 {
		set valorMatriz1 = matrizA.matriz.%Get(m).%Get(k)
		set valorMatriz2 = matrizB.matriz.%Get(k).%Get(l)
		set soma=soma+(valorMatriz1 * valorMatriz2)
	}
	quit soma
}

ClassMethod MapSomarMatriz(valorLinha As %Float, m As %Integer, l As %Integer, matrizA As Matriz, matrizB As Matriz) As %Float
{
	quit matrizA.matriz.%Get(m).%Get(l)+matrizB.matriz.%Get(m).%Get(l)
}

ClassMethod MapSubtrairMatriz(valorLinha As %Float, m As %Integer, l As %Integer, matrizA As Matriz, matrizB As Matriz) As %Float
{
	quit matrizA.matriz.%Get(m).%Get(l)-matrizB.matriz.%Get(m).%Get(l)
}

Storage Default
{
<Data name="MatrizDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>linhas</Value>
</Value>
<Value name="3">
<Value>colunas</Value>
</Value>
<Value name="4">
<Value>matriz</Value>
</Value>
</Data>
<DataLocation>^MatrizD</DataLocation>
<DefaultData>MatrizDefaultData</DefaultData>
<IdLocation>^MatrizD</IdLocation>
<IndexLocation>^MatrizI</IndexLocation>
<StreamLocation>^MatrizS</StreamLocation>
<Type>%Library.CacheStorage</Type>
}

}
