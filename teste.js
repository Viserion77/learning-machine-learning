setInterval(() => document.location.reload(), 100000);

const contente = document.getElementById('contente');
contente.innerHTML = Date.now();

var redeNeural = new RedeNeural(2, 3, 1);

let dataset = {
  inputs: [
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0],
  ],
  saidas: [
    [0],
    [1],
    [1],
    [0],
  ],
};

let treinos = 100000;

for (let index = 0; index < treinos; index++) {
  var randoValue = Math.floor(Math.random(4));
  redeNeural.treino(dataset.inputs[randoValue], dataset.saidas[randoValue]);
}
console.log('0.04>', redeNeural.predict([0, 0]));
console.log('0.04>', redeNeural.predict([1, 1]));
console.log('0.96<', redeNeural.predict([1, 0]));
console.log('0.96<', redeNeural.predict([0, 1]));
//while (redeNeural.predict([0, 0]) > 0.04 || redeNeural.predict([1, 0]) < 0.96 && treinos < 50000) {
//  var randoValue = Math.floor(Math.random(4));
//  redeNeural.treino(dataset.inputs[randoValue], dataset.saidas[randoValue]);
//  treinos++;
//}

contente.innerHTML = treinos