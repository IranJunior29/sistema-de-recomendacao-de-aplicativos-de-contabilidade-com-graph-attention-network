# Imports
import pandas as pd
import numpy as np
import sklearn
import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import to_undirected
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Classe do modelo
class RecommenderGAT(torch.nn.Module):

    # Método construtor
    def __init__(self, num_nodes, hidden_dim=32, output_dim=1, heads=2):
        super(RecommenderGAT, self).__init__()

        # Primeira camada convolucional de atenção do grafo (GATConv)
        self.conv1 = GATConv(num_nodes, hidden_dim, heads=heads)

        # Segunda camada convolucional de atenção do grafo (GATConv)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads)

        # Camada linear (fully connected) para produzir a saída final
        self.linear = torch.nn.Linear(hidden_dim * heads, output_dim)

    # Método forward
    def forward(self, data):
        # Extração dos atributos dos nós e da matriz de índices de arestas do objeto Data
        x, edge_index = data.x, data.edge_index

        # Aplicação da primeira camada GAT (self.conv1) seguida de uma função de ativação
        x = self.conv1(x, edge_index)

        # Exponential Linear Unit (ELU)
        x = F.elu(x)

        # Aplicação de dropout para regularização (com probabilidade de 0,5)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)

        x = F.elu(x)

        # Agregação global para criar uma representação geral do grafo
        x = global_mean_pool(x, torch.zeros(1, dtype=torch.long))

        # Camada linear (self.linear) para gerar a saída final
        x = self.linear(x)

        return x

if __name__ == '__main__':

    ''' Verificando o Ambiente de Desenvolvimento '''

    # Verifica se uma GPU está disponível e define o dispositivo apropriado
    processing_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define o device (GPU ou CPU)
    device = torch.device(processing_device)
    print(device)

    ''' Preparando o Conjunto de Dados '''

    # Aqui, você deve carregar seus próprios dados de usuários, aplicativos (ou qualquer outra coisa) e avaliações.
    # Como exemplo, criaremos um dataset fictício.

    # Dataframe de usuários
    users = pd.DataFrame({"user_id": [0, 1, 2, 3]})

    # Dataframe de apps
    apps = pd.DataFrame({"app_id": [0, 1, 2]})

    # Dataframe de avaliações (ratings)
    ratings = pd.DataFrame({"user_id": [0, 1, 1, 2, 3],
                            "app_id": [0, 0, 1, 2, 2],
                            "rating": [4, 5, 3, 2, 4]})

    # Converta os IDs dos aplicativos para evitar confução com os IDs dos usuários
    ratings["app_id"] += users.shape[0] - 1

    ''' Pré-Processamento dos Dados no Formato de Grafo '''

    # Divide os dados em conjuntos de treinamento e teste
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

    # Prepara os nodes de treino
    train_source_nodes = torch.tensor(train_ratings["user_id"].values, dtype=torch.long)
    train_target_nodes = torch.tensor(train_ratings["app_id"].values, dtype=torch.long)

    # Prepara os edges (arestas) de treino
    train_edge_index = torch.stack([train_source_nodes, train_target_nodes], dim=0)
    train_edge_index = to_undirected(train_edge_index)
    train_edge_attr = torch.tensor(train_ratings["rating"].values, dtype=torch.float32).view(-1, 1)

    # Repete o processo anterior para os dados de teste
    test_source_nodes = torch.tensor(test_ratings["user_id"].values, dtype=torch.long)
    test_target_nodes = torch.tensor(test_ratings["app_id"].values, dtype=torch.long)

    test_edge_index = torch.stack([test_source_nodes, test_target_nodes], dim=0)
    test_edge_index = to_undirected(test_edge_index)
    test_edge_attr = torch.tensor(test_ratings["rating"].values, dtype=torch.float32).view(-1, 1)

    # Números de nodes
    num_nodes = users.shape[0] + apps.shape[0]

    #Cria os atributos de aresta do conjunto de treinamento
    train_data = Data(x=torch.eye(num_nodes, dtype=torch.float32),
                      edge_index=train_edge_index,
                      edge_attr=train_edge_attr,
                      y=train_edge_attr)

    # Cria os atributos de aresta do conjunto de teste
    test_data = Data(x=torch.eye(num_nodes, dtype=torch.float32),
                     edge_index=test_edge_index,
                     edge_attr=test_edge_attr,
                     y=test_edge_attr)

    # Cria os dataloaders (o que é requerido pelo PyTorch)
    train_data_loader = DataLoader([train_data], batch_size=1)
    test_data_loader = DataLoader([test_data], batch_size=1)

    ''' Construção do Modelo '''

    # Cria instância do modelo
    model = RecommenderGAT(num_nodes=num_nodes)

    # Função de erro
    loss_fn = torch.nn.MSELoss()

    # Otimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Número de épocas
    num_epochs = 100

    ''' Treinamento e Avaliação do Modelo '''

    # Coloca o modelo em modo de treino
    model.train()

    # Loop de treino
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_data_loader:
            optimizer.zero_grad()
            predictions = model(batch)
            loss = loss_fn(predictions, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch + 1}, Loss: {total_loss / len(train_data_loader)}')


    # Coloca o modelo em modo de avaliação
    model.eval()

    # Faz previsões no conjunto de teste
    with torch.no_grad():
        test_predictions = model(test_data).numpy()

    # Calcula o RMSE comparando as previsões com as avaliações reais
    rmse = np.sqrt(np.mean((test_predictions.flatten() - test_data.y.numpy().flatten()) ** 2))
    print(f'RMSE no conjunto de teste: {rmse}')

    ''' Deploy e Teste do Sistema de Recomendação '''

    # Faz a previsão com base nos dados de teste
    with torch.no_grad():
        previsao = model(test_data).numpy()

    print(previsao)

    # Cria um DataFrame com as previsões
    df_previsoes = pd.DataFrame({
        "user_id": test_ratings["user_id"].values,
        "app_id": test_ratings["app_id"].values,
        "predicted_rating": np.round(previsao.flatten(), 0)
    })

    # Exibe a previsão
    print(df_previsoes)

    # Itera sobre as linhas do DataFrame
    for index, row in df_previsoes.iterrows():
        if row["predicted_rating"] >= 3:
            print(f"Recomendamos o aplicativo {row['app_id']} para o usuário {row['user_id']}.")
        else:
            print(f"Não recomendamos o aplicativo {row['app_id']} para o usuário {row['user_id']}.")
