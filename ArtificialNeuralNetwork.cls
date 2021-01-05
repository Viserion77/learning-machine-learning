/// Em ciência da computação e campos relacionados, redes neurais artificiais são modelos computacionais inspirados pelo sistema nervoso central de um animal que são capazes de realizar o aprendizado de máquina bem como o reconhecimento de padrões.
Class ArtificialNeuralNetwork Extends %Persistent [ ClassType = persistent, ProcedureBlock ]
{

Parameter constanteEuler As %Float = 2.71828182849045235360287471352662497757;

Property neuroniosEntrada As %Integer;

Property neuroniosOcultos As %Integer;

Property neuroniosSaida As %Integer;

Property biasEntradaOculta As Matriz;

Property biasOcultaSaida As Matriz;

Property pesosEntradaOculta As Matriz;

Property pesosOcultaSaida As Matriz;

Property areaAprendizado As %Float;

Method %OnNew(neuroniosEntrada As %Integer = 1, neuroniosOcultos As %Integer = 1, neuroniosSaida As %Integer = 1) As %Status
{
	set ..neuroniosEntrada=neuroniosEntrada
	set ..neuroniosOcultos=neuroniosOcultos
	set ..neuroniosSaida=neuroniosSaida
	
	set matrizBiasEntradaOculta=##class(Matriz).%New(..neuroniosOcultos,1,,1)
	set ..biasEntradaOculta=matrizBiasEntradaOculta
	
	set matrizBiasOcultaSaida=##class(Matriz).%New(..neuroniosSaida,1,,1)
	set ..biasOcultaSaida=matrizBiasOcultaSaida
	
	set matrizPesosEntradaOculta=##class(Matriz).%New(..neuroniosOcultos,..neuroniosEntrada,,1)
	set ..pesosEntradaOculta=matrizPesosEntradaOculta
	
	set matrizPesosOcultaSaida=##class(Matriz).%New(..neuroniosSaida,..neuroniosOcultos,,1)
	set ..pesosOcultaSaida=matrizPesosOcultaSaida
	
	set ..areaAprendizado=0.1
	
	quit $$$OK
}

// w !,##class(ArtificialNeuralNetwork).Teste()

ClassMethod Teste() As %Status
{
	set sc=$$$OK
	set ann=##class(ArtificialNeuralNetwork).%New(2,3,1)

	// XOR Problem
	set dataset = {
		"inputs":
			[[1, 1],
			[1, 0],
			[0, 1],
			[0, 0]],
		"outputs":
			[[0],
			[1],
			[1],
			[0]]
	}
	for i=0:1:50000 {
		w #!,"Treino "_i_" de 50000"
		set index=$random(4)
		do ann.Treinar(dataset.inputs.%Get(index), dataset.outputs.%Get(index))
	}
	write !,"Teste Finalizado"
	write !,"Teste com `write ann.Predict([0, 0]).%Get(0)` "_ann.Predict([0, 0]).%Get(0)_" Deveria ser 0"
	write !,"Teste com `write ann.Predict([1, 1]).%Get(0)` "_ann.Predict([1, 1]).%Get(0)_" Deveria ser 0"
	write !,"Teste com `write ann.Predict([0, 1]).%Get(0)` "_ann.Predict([0, 1]).%Get(0)_" Deveria ser 1"
	write !,"Teste com `write ann.Predict([1, 0]).%Get(0)` "_ann.Predict([1, 0]).%Get(0)_" Deveria ser 1"
	break
	quit sc
}

// w !,##class(ArtificialNeuralNetwork).Teste2()

ClassMethod Teste2() As %Status
{
	set sc=$$$OK
	//                                5 entradas / 77 neuronios internos / 1 saida
	set ann=##class(ArtificialNeuralNetwork).%New(5,77,1)

	// Dragon Or Aguia?
	// [Pode voar?, possui dentes?, possui penas?, possui braços?, possui assas?]
	// [1] = dragon
	// [0] = Aguia
	set dataset = {
		"inputs":
			[[1, 0, 1, 0, 1],
			[1, 1, 1, 0, 1],
			[0, 1, 0, 1, 1],
			[1, 0, 1, 1, 1],
			[1, 1, 1, 0, 1],
			[0, 1, 0, 1, 1],
			[1, 0, 1, 1, 1]],
		"outputs":
			[[0],
			[1],
			[1],
			[0],
			[1],
			[1],
			[0]]
	}
	for i=0:1:500 {
		w #!,"Treino "_i_" de 500"
		set index=$random(4)
		do ann.Treinar(dataset.inputs.%Get(index), dataset.outputs.%Get(index))
	}
	write !,"Teste Finalizado"
	write !,"Teste com `write ann.Predict([1, 0, 1, 0, 1]).%Get(0)` "_ann.Predict([1, 0, 1, 0, 1]).%Get(0)_" Deveria ser 0"
	write !,"Teste com `write ann.Predict([1, 0, 1, 0, 1]).%Get(0)` "_ann.Predict([1, 0, 1, 0, 1]).%Get(0)_" Deveria ser 0"
	break
	quit sc
}

Method Treinar(arrayEntradaDados As %Library.DynamicArray, arrayAlvo As %Library.DynamicArray) As %Status
{
	set matrizEntrada=##class(Matriz).ArrayToMatriz(arrayEntradaDados)
	set matrizOculta=##class(Matriz).Multiplicar(..pesosEntradaOculta, matrizEntrada)
	set matrizOculta=##class(Matriz).Adicionar(matrizOculta, ..biasEntradaOculta)
	set matrizOculta=matrizOculta.Map(..%ClassName(1),"Sigmoid")
	
	set matrizSaida=##class(Matriz).Multiplicar(..pesosOcultaSaida, matrizOculta)
	set matrizSaida=##class(Matriz).Adicionar(matrizSaida, ..biasOcultaSaida)
	set matrizSaida=matrizSaida.Map(..%ClassName(1),"Sigmoid")
	
	set matrizEspectativa=##class(Matriz).ArrayToMatriz(arrayAlvo)
	set matrizErroSaida=##class(Matriz).Subtrair(matrizEspectativa, matrizSaida)
	set matrizSaidaDerivada=matrizSaida.Map(..%ClassName(1),"SigmoidDerivada")
	set matrizOcultaTransposta=##class(Matriz).Transpor(matrizOculta)
	
	set matrizGradienteEntrada=##class(Matriz).Hadamard(matrizSaidaDerivada, matrizErroSaida)
	set matrizGradienteEntrada=##class(Matriz).EscalarMultiplicar(matrizGradienteEntrada, ..areaAprendizado)
	
	set ..biasOcultaSaida=##class(Matriz).Adicionar(..biasOcultaSaida, matrizGradienteEntrada)
	set matrizPesosOcultaSaidaDelta=##class(Matriz).Multiplicar(matrizGradienteEntrada, matrizOcultaTransposta)
	set ..pesosOcultaSaida=##class(Matriz).Adicionar(..pesosOcultaSaida, matrizPesosOcultaSaidaDelta)
	
	set matrizPesosOcultaSaidaTransposto=##class(Matriz).Transpor(..pesosOcultaSaida)
	set matrizErroOculta=##class(Matriz).Multiplicar(matrizPesosOcultaSaidaTransposto, matrizErroSaida)
	set matrizOcultaDerivada=matrizOculta.Map(..%ClassName(1),"SigmoidDerivada")
	set matrizEntradaTransposta=##class(Matriz).Transpor(matrizEntrada)
	
	set matrizGradienteOculta=##class(Matriz).Hadamard(matrizOcultaDerivada, matrizErroOculta)
	set matrizGradienteOculta=##class(Matriz).EscalarMultiplicar(matrizGradienteOculta, ..areaAprendizado)
	
	set ..biasEntradaOculta=##class(Matriz).Adicionar(..biasEntradaOculta, matrizGradienteOculta)
	set matrizPesosEntradaOcultaDelta=##class(Matriz).Multiplicar(matrizGradienteOculta, matrizEntradaTransposta)
	set ..pesosEntradaOculta=##class(Matriz).Adicionar(..pesosEntradaOculta, matrizPesosEntradaOcultaDelta)
	
	quit $$$OK
}

Method Predict(arrayEntradaDados As %Library.DynamicArray) As %Library.DynamicArray
{
	set matrizEntrada=##class(Matriz).ArrayToMatriz(arrayEntradaDados)
	set matrizOculta=##class(Matriz).Multiplicar(..pesosEntradaOculta, matrizEntrada)
	set matrizOculta=##class(Matriz).Adicionar(matrizOculta, ..biasEntradaOculta)
	set matrizOculta=matrizOculta.Map(..%ClassName(1),"Sigmoid")
	
	set matrizSaida=##class(Matriz).Multiplicar(..pesosOcultaSaida, matrizOculta)
	set matrizSaida=##class(Matriz).Adicionar(matrizSaida, ..biasOcultaSaida)
	set matrizSaida=matrizSaida.Map(..%ClassName(1),"Sigmoid")
	set arraySaida=##class(Matriz).MatrizToArray(matrizSaida)
	
	quit arraySaida
}

ClassMethod Sigmoid(x, a1, a2, a3, a4) As %Float
{
	quit 1/(1+##class(ArtificialNeuralNetwork).Exp(-x))
}

ClassMethod SigmoidDerivada(x, a1, a2, a3, a4) As %Float
{
	quit x*(1-x)
}

ClassMethod Exp(x) As %Float
{
	quit ##class(ArtificialNeuralNetwork).#constanteEuler**x
}

Storage Default
{
<Data name="ArtificialNeuralNetworkDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>neuroniosEntrada</Value>
</Value>
<Value name="3">
<Value>neuroniosOcultos</Value>
</Value>
<Value name="4">
<Value>neuroniosSaida</Value>
</Value>
<Value name="5">
<Value>biasEntradaOculta</Value>
</Value>
<Value name="6">
<Value>biasOcultaSaida</Value>
</Value>
<Value name="7">
<Value>pesosEntradaOculta</Value>
</Value>
<Value name="8">
<Value>pesosOcultaSaida</Value>
</Value>
<Value name="9">
<Value>areaAprendizado</Value>
</Value>
</Data>
<DataLocation>^ArtificialNDFE0D</DataLocation>
<DefaultData>ArtificialNeuralNetworkDefaultData</DefaultData>
<IdLocation>^ArtificialNDFE0D</IdLocation>
<IndexLocation>^ArtificialNDFE0I</IndexLocation>
<StreamLocation>^ArtificialNDFE0S</StreamLocation>
<Type>%Library.CacheStorage</Type>
}

}
