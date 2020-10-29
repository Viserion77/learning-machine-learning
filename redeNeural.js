function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivada(x){
    return x * (1-x); 
}

class RedeNeural {
    constructor(neuroniosEntrada, neuroniosOcultos, neuroniosSaida) {
        this.neuroniosEntrada = neuroniosEntrada;
        this.neuroniosOcultos = neuroniosOcultos;
        this.neuroniosSaida = neuroniosSaida;

        this.biasEntradaOculta = new Matriz(this.neuroniosOcultos, 1);
        this.biasEntradaOculta.aleatorizar();
        this.biasOcultaSaida = new Matriz(this.neuroniosSaida, 1);
        this.biasOcultaSaida.aleatorizar();

        this.pesosEntradaOculta = new Matriz(this.neuroniosOcultos, this.neuroniosEntrada);
        this.pesosEntradaOculta.aleatorizar()

        this.pesosOcultaSaida = new Matriz(this.neuroniosSaida, this.neuroniosOcultos)
        this.pesosOcultaSaida.aleatorizar()

        this.areaAprendizado = 0.1;
    }

    treino(arrayEntrada,arrayResposta) {
        // INPUT -> HIDDEN
        let entrada = Matriz.arrayToMatriz(arrayEntrada);
        let oculta = Matriz.multiplicar(this.pesosEntradaOculta, entrada);
        oculta = Matriz.adicionar(oculta, this.biasEntradaOculta);

        oculta.mapear(sigmoid)

        // HIDDEN -> OUTPUT
        // d(Sigmoid) = Output * (1- Output)
        let saida = Matriz.multiplicar(this.pesosOcultaSaida, oculta);
        saida = Matriz.adicionar(saida, this.biasOcultaSaida);
        saida.mapear(sigmoid);

        // BACKPROPAGATION

        // OUTPUT -> HIDDEN
        let expectativa = Matriz.arrayToMatriz(arrayResposta);
        let erroSaida = Matriz.subtrair(expectativa,saida);
        let saidaDerivada = Matriz.mapear(saida,sigmoidDerivada);
        let ocultaTransposta = Matriz.transpor(oculta);

        let gradienteEntrada = Matriz.hadamard(saidaDerivada,erroSaida);
        gradienteEntrada = Matriz.escalarMultiplicar(gradienteEntrada,this.areaAprendizado);
        
        // Adjust Bias O->H
        this.biasOcultaSaida = Matriz.adicionar(this.biasOcultaSaida, gradienteEntrada);
        // Adjust Weigths O->H
        let pesosOcultaSaidaDeltas = Matriz.multiplicar(gradienteEntrada,ocultaTransposta);
        this.pesosOcultaSaida = Matriz.adicionar(this.pesosOcultaSaida,pesosOcultaSaidaDeltas);

        // HIDDEN -> INPUT
        let pesosOcultaSaidaTransposto = Matriz.transpor(this.pesosOcultaSaida);
        let erroOculta = Matriz.multiplicar(pesosOcultaSaidaTransposto,erroSaida);
        let ocultaDerivada = Matriz.mapear(oculta,sigmoidDerivada);
        let entradaTransposta = Matriz.transpor(entrada);

        let gradienteOculta = Matriz.hadamard(ocultaDerivada,erroOculta);
        gradienteOculta = Matriz.escalarMultiplicar(gradienteOculta, this.areaAprendizado);

        // Adjust Bias O->H
        this.biasEntradaOculta = Matriz.adicionar(this.biasEntradaOculta, gradienteOculta);
        // Adjust Weigths H->I
        let pesosEntradaOculta_deltas = Matriz.multiplicar(gradienteOculta, entradaTransposta);
        this.pesosEntradaOculta = Matriz.adicionar(this.pesosEntradaOculta, pesosEntradaOculta_deltas);
    }

    predict(arrayEntrada){
        // INPUT -> HIDDEN
        let entrada = Matriz.arrayToMatriz(arrayEntrada);

        let oculta = Matriz.multiplicar(this.pesosEntradaOculta, entrada);
        oculta = Matriz.adicionar(oculta, this.biasEntradaOculta);

        oculta.mapear(sigmoid)

        // HIDDEN -> OUTPUT
        let saida = Matriz.multiplicar(this.pesosOcultaSaida, oculta);
        saida = Matriz.adicionar(saida, this.biasOcultaSaida);
        saida.mapear(sigmoid);
        saida = Matriz.MatrizToArray(saida);

        return saida;
    }
}