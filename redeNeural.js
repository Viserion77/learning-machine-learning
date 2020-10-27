function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
function derivadaSigmoid(x) {
    return x * (1 - x);
}
class RedeNeural {
    constructor(numNeuroniosEntrada, numNeuroniosOcultos, numNeuroniosSaida) {
        this.numNeuroniosEntrada = numNeuroniosEntrada;
        this.numNeuroniosOcultos = numNeuroniosOcultos;
        this.numNeuroniosSaida = numNeuroniosSaida;

        this.biasEntradaToOculta = new Matriz(this.numNeuroniosOcultos, 1);
        this.biasEntradaToOculta.randomizarValores();
        this.biasOcultaToSaida = new Matriz(this.numNeuroniosSaida, 1);
        this.biasOcultaToSaida.randomizarValores();

        this.pessosEntradaToOculta = new Matriz(this.numNeuroniosOcultos, this.numNeuroniosEntrada);
        this.pessosEntradaToOculta.randomizarValores();

        this.pessosOcultaToSaida = new Matriz(this.numNeuroniosSaida, this.numNeuroniosOcultos);
        this.pessosOcultaToSaida.randomizarValores();
        this.areaAprendizado = 0.1;
    }

    treino(ArrayEntrada, alvo) {
        let entrada = Matriz.arrayToMatriz(ArrayEntrada);
        let camadaOculta = Matriz.multiplicar(this.pessosEntradaToOculta, entrada);

        camadaOculta = Matriz.adicionar(camadaOculta, this.biasEntradaToOculta);
        camadaOculta.mapear(sigmoid);

        let camadaSaida = Matriz.multiplicar(this.pessosOcultaToSaida, camadaOculta);
        camadaSaida = Matriz.adicionar(camadaSaida, this.biasOcultaToSaida);
        camadaSaida.mapear(sigmoid);

        let esperado = Matriz.arrayToMatriz(alvo);
        let erroSaida = Matriz.subtrair(esperado, camadaSaida);
        let derivadaSaida = Matriz.mapear(camadaSaida, derivadaSigmoid);

        let camadaOcultaTransposta = Matriz.transpose(camadaOculta);

        let gradiente = Matriz.hadamard(erroSaida, derivadaSaida);

        gradiente = Matriz.escalarMultiplicar(gradiente, this.areaAprendizado);
        this.biasOcultaToSaida = Matriz.adicionar(this.biasOcultaToSaida, gradiente);

        let pessosOcultaToSaidaDeltas = Matriz.multiplicar(gradiente, camadaOcultaTransposta);

        this.pessosOcultaToSaida = Matriz.adicionar(this.pessosOcultaToSaida, pessosOcultaToSaidaDeltas);

        let pessosEntradaToOcultaTransposto = Matriz.transpose(this.pessosOcultaToSaida);

        let erroOculta = Matriz.multiplicar(pessosEntradaToOcultaTransposto, erroSaida);
        let derivadaOculta = Matriz.mapear(camadaOculta, derivadaSigmoid);
        let entradaTransposta = Matriz.transpose(entrada);

        let gradienteOculta = Matriz.hadamard(erroOculta, derivadaOculta);

        gradienteOculta = Matriz.escalarMultiplicar(gradienteOculta, this.areaAprendizado);
        this.biasEntradaToOculta = Matriz.adicionar(this.biasEntradaToOculta, gradienteOculta);

        let pessosEntradaToOcultaDelta = Matriz.multiplicar(gradienteOculta, entradaTransposta);

        this.pessosEntradaToOculta = Matriz.adicionar(this.pessosEntradaToOculta, pessosEntradaToOcultaDelta);

    }
    predict(ArrayEntrada) {
        let entrada = Matriz.arrayToMatriz(ArrayEntrada);
        let camadaOculta = Matriz.multiplicar(this.pessosEntradaToOculta, entrada);
        camadaOculta = Matriz.adicionar(camadaOculta, this.biasEntradaToOculta);
        camadaOculta.mapear(sigmoid);

        let camadaSaida = Matriz.multiplicar(this.pessosOcultaToSaida, camadaOculta);
        camadaSaida = Matriz.adicionar(camadaSaida, this.biasOcultaToSaida);
        camadaSaida.mapear(sigmoid);

        camadaSaida = Matriz.MatrizToArray(camadaSaida);
        return camadaSaida;
    }
}