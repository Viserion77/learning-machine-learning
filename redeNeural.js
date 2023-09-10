function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function step (x) {
    return (x >= 1) ? 1 : 0;
}

function hyperbolicTangent(x) {
    return (Math.exp(x)-Math.exp(-x))/(Math.exp(x)+Math.exp(-x))
}

function relu(x) {
    return (x >= 0) ? x : 0;
}

function linear(x) {
    return x;
}

function softmax(x) {
    return Math.exp(x) / Math.exp(x).reduce((a, b) => a + b, 0);
}

function sigmoidDerivada(x) {
    return x * (1 - x);
}

class RedeNeural {
    constructor(neuroniosEntrada, neuroniosOcultos, a, b, neuroniosSaida) {
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

    treino(arrayEntrada, arrayResposta) {
        const entrada = Matriz.arrayToMatriz(arrayEntrada)
        const oculta = Matriz.adicionar(Matriz.multiplicar(this.pesosEntradaOculta, entrada), this.biasEntradaOculta).mapear(sigmoid);
        const saida = Matriz.adicionar(Matriz.multiplicar(this.pesosOcultaSaida, oculta), this.biasOcultaSaida).mapear(sigmoid);

        // BACKPROPAGATION

        const erroSaida = Matriz.subtrair(Matriz.arrayToMatriz(arrayResposta), saida);
        const gradienteEntrada = Matriz.escalarMultiplicar(Matriz.hadamard(Matriz.mapear(saida, sigmoidDerivada), erroSaida), this.areaAprendizado);
        this.biasOcultaSaida = Matriz.adicionar(this.biasOcultaSaida, gradienteEntrada);
        this.pesosOcultaSaida = Matriz.adicionar(this.pesosOcultaSaida, Matriz.multiplicar(gradienteEntrada, Matriz.transpor(oculta)));
        const gradienteOculta = Matriz.escalarMultiplicar(Matriz.hadamard(Matriz.mapear(oculta, sigmoidDerivada), Matriz.multiplicar(Matriz.transpor(this.pesosOcultaSaida), erroSaida)), this.areaAprendizado);
        this.biasEntradaOculta = Matriz.adicionar(this.biasEntradaOculta, gradienteOculta);
        this.pesosEntradaOculta = Matriz.adicionar(this.pesosEntradaOculta, Matriz.multiplicar(gradienteOculta, Matriz.transpor(entrada)));
    }

    predict(arrayEntrada) {
        return Matriz.MatrizToArray(
            Matriz.adicionar(
                Matriz.multiplicar(
                    this.pesosOcultaSaida,
                    Matriz.adicionar(
                        Matriz.multiplicar(this.pesosEntradaOculta, Matriz.arrayToMatriz(arrayEntrada)), this.biasEntradaOculta).mapear(sigmoid)), this.biasOcultaSaida).mapear(sigmoid));
    }
}