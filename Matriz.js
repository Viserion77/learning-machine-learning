class Matriz{
    constructor(linhas, colunas){
        this.linhas= linhas;
        this.colunas = colunas;

        this.conteudo = [];

        for(let i=0; i<linhas; i++){
            let array = [];
            for (let j = 0; j < colunas; j++) {
                array.push(0);
            }
            this.conteudo.push(array);
        }
    }

    static mapear(matriz1, matriz2, funcao){
        let matriz = new matriz(matriz1.linhas, matriz2.linhas);

        matriz.conteudo = matriz.conteudo.map((array,i)=>{
            return array.map((numero,j)=>{
                return funcao(numero,i,j);
            })
        })
        return matriz;
    }

    mapear(funcao){
        this.conteudo = this.conteudo.map((array,i)=>{
            return array.map((numero,j)=>{
                return funcao(numero,i,j);
            })
        })
        return this;
    }

    static adicionar(matriz1, matriz2){
        var matriz = new Matriz(matriz1.linhas, matriz1.colunas);

        matriz.mapear((elemento, i, j)=>{
            return matriz1.conteudo[i][j] + matriz2.conteudo[i][j];
        });

        return matriz
    }

    static multiplicar(matriz1, matriz2){
        var matriz = new Matriz(matriz1.linhas, matriz2.colunas);
        
        matriz.mapear((elemento, i, j)=>{
            let soma=0
            for (let k=0; k<matriz1.colunas; k++){
                let valorMatriz1=matriz1.conteudo[i][k];
                let valorMatriz2=matriz2.conteudo[k][j];
                soma += valorMatriz1*valorMatriz2;
            }
            return soma
        });

        return matriz
    }

    randomizarValores(){
        this.mapear((elemento,i,j)=>{
            return Math.random();
        });
        return this;
    }

    print(){
        console.log(this)
        console.table(this.conteudo)
    }

    static arrayToMatriz(array){
        let matriz = new Matriz(array.length,1)
        matriz.mapear((elemento,i,j)=>{
            return array[i];
        });
        return matriz;
    }
}