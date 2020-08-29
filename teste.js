setInterval(()=>document.location.reload(),20000);

const contente=document.getElementById('contente');
contente.innerHTML=Date.now();

var redeNeural = new RedeNeural(1,3,1)
redeNeural.feedforward([1,2])

contente.innerHTML=redeNeural