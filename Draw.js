class Draw {
    static matrizToTableHtml(matriz) {
        var tableHtml = '<table border="1">';
        for (let i = 0; i < matriz.linhas; i++) {
            tableHtml += "<tr>";
            for (let j = 0; j < matriz.colunas; j++) {
                tableHtml += "<td>";
                tableHtml += matriz.conteudo[i][j];
                tableHtml += "</td>";
            }
            tableHtml += "</tr>";
        }
        tableHtml += "</table>";
        return tableHtml;
    }
}