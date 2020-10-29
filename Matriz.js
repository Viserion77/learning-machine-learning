class Matriz {
    constructor(linhas, colunas) {
        this.linhas = linhas;
        this.colunas = colunas;

        this.conteudo = [];

        for (let i = 0; i < linhas; i++) {
            let array = [];
            for (let j = 0; j < colunas; j++) {
                array.push(0);
            }
            this.conteudo.push(array);
        }
    }
    static mapear(matriz1, funcao) {
        let matriz = new Matriz(matriz1.linhas, matriz1.colunas);

        matriz.conteudo = matriz1.conteudo.map((array, i) => {
            return array.map((numero, j) => {
                return funcao(numero, i, j);
            })
        })
        return matriz;
    }

    mapear(funcao) {
        this.conteudo = this.conteudo.map((array, i) => {
            return array.map((numero, j) => {
                return funcao(numero, i, j);
            })
        })
        return this;
    }

    static hadamard(matriz1, matriz2) {
        var matriz = new Matriz(matriz1.linhas, matriz1.colunas);

        matriz.mapear((numero, i, j) => {
            return matriz1.conteudo[i][j] * matriz2.conteudo[i][j];
        });
        return matriz;
    }

    static escalarMultiplicar(matriz1, escalar) {
        var matriz = new Matriz(matriz1.linhas, matriz1.colunas);

        matriz.mapear((numero, i, j) => {
            return matriz1.conteudo[i][j] * escalar;
        });
        return matriz;
    }

    static transpor(matriz1) {
        var matriz = new Matriz(matriz1.colunas, matriz1.linhas);
        matriz.mapear((numero, i, j) => {
            return matriz1.conteudo[j][i];
        });
        return matriz;
    }

    static subtrair(matriz1, matriz2) {
        var matriz = new Matriz(matriz1.linhas, matriz1.colunas);

        matriz.mapear((numero, i, j) => {
            return matriz1.conteudo[i][j] - matriz2.conteudo[i][j]
        });

        return matriz;
    }

    static adicionar(matriz1, matriz2) {
        var matriz = new Matriz(matriz1.linhas, matriz1.colunas);

        matriz.mapear((numero, i, j) => {
            return matriz1.conteudo[i][j] + matriz2.conteudo[i][j]
        });

        return matriz;
    }

    static multiplicar(matriz1, matriz2) {
        var matriz = new Matriz(matriz1.linhas, matriz2.colunas);

        matriz.mapear((numero, i, j) => {
            let soma = 0
            for (let k = 0; k < matriz1.colunas; k++) {
                let valorMatriz1 = matriz1.conteudo[i][k];
                let valorMatriz2 = matriz2.conteudo[k][j];
                soma += valorMatriz1 * valorMatriz2;
            }
            return soma
        });

        return matriz
    }

    aleatorizar() {
        this.mapear((elemento, i, j) => {
            return Math.random() * 2 - 1;
        });
    }

    static arrayToMatriz(array) {
        let matriz = new Matriz(array.length, 1);
        matriz.mapear((elemento, i, j) => {
            return array[i];
        });
        return matriz;
    }

    static MatrizToArray(matriz) {
        let array = [];
        matriz.mapear((elemento) => {
            array.push(elemento);
        })
        return array;
    }
}