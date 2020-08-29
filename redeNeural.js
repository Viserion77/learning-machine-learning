function sigmoid(x){
    return 1/(1+Math.exp(-x));
}
class RedeNeural{
    constructor(numNeuroniosEntrada, numNeuroniosOcultos, numNeuroniosSaida){
        this.numNeuroniosEntrada = numNeuroniosEntrada;
        this.numNeuroniosOcultos = numNeuroniosOcultos;
        this.numNeuroniosSaida = numNeuroniosSaida;

        this.biasEntradaToOculta = new Matriz(this.numNeuroniosOcultos,1);
        this.biasEntradaToOculta.randomizarValores();
        this.biasOcultaToSaida = new Matriz(this.numNeuroniosSaida,1);
        this.biasOcultaToSaida.randomizarValores();

        this.pessosEntradaToOculta = new Matriz(this.numNeuroniosOcultos, this.numNeuroniosEntrada);
        this.pessosEntradaToOculta.randomizarValores();

        this.pessosOcultaToSaida = new Matriz(this.numNeuroniosSaida, this.numNeuroniosOcultos);
        this.pessosOcultaToSaida.randomizarValores();
    }

    feedforward(ArrayEntrada){
        let entrada= Matriz.arrayToMatriz(ArrayEntrada);
        let camadaOculta = Matriz.multiplicar(this.pessosEntradaToOculta, entrada);
        camadaOculta = Matriz.adicionar(camadaOculta, this.biasEntradaToOculta);
        camadaOculta.mapear(sigmoid)

        let camadaSaida = Matriz.multiplicar(this.pessosOcultaToSaida, camadaOculta);
        camadaSaida = Matriz.adicionar(camadaSaida, this.biasOcultaToSaida);
        camadaSaida.mapear(sigmoid)

        camadaSaida.print()
    }
}