## 🧠 Sistema de Reconhecimento Facial com OpenCV (Cadastro + Reconhecimento Facial)
#### Este projeto implementa um sistema completo de cadastro e reconhecimento facial utilizando Python, OpenCV e o algoritmo LBPH (Local Binary Patterns Histogram).

Funcionalidades
- 👤 Cadastrar novos usuários via webcam.
- 📸 Capturar automaticamente 150 imagens por pessoa.
- 🧠 Treinar um modelo de reconhecimento facial com LBPH.
- 🎥 Reconhecer múltiplas pessoas simultaneamente ao vivo.
- 🔄 Iniciar automaticamente em modo "reconhecimento", caso rostos já estejam cadastrados.
- 💾 Armazenar datasets por usuário, em pastas separadas.

### 📂 Estrutura do Projeto
```
FaceRec/
│
├── faces_dataset/nome_usuario    # Pastas criadas automaticamente após cadastro.
├── main.py           # Código principal do sistema.
└── README.md
```

### 📦 Dependências
```
bash
pip install opencv-contrib-python numpy
```
### ⚠️ Importante:
Use opencv-contrib-python (não apenas opencv-python), pois o módulo cv2.face só existe na versão contrib.

## Execução (VsCode)
Execute o programa pelo terminal:
```
python main.py
```

#### Se existirem usuários cadastrados, sistema treina automaticamente e inicia direto no reconhecimento ao vivo.

#### Se NÃO existirem usuários cadastrados, siga o fluxo abaixo:
```
Aperte C para cadastrar o rosto

Ao pressionar C:
    1. Digite o nome
    2. O sistema captura 150 imagens
    3. Treina o modelo
    4. Inicia o reconhecimento
    
Pressione ESC para sair em qualquer tela.
```

### ⚙️ Parâmetros Importantes
| Parâmetro        | Valor             | Descrição                                          |
| ---------------- | ----------------- | -------------------------------------------------- |
| `DATASET_PATH`   | `"faces_dataset"` | Diretório onde cada usuário terá sua pasta         |
| `CAPTURE_COUNT`  | `150`             | Número de imagens capturadas no cadastro           |
| `LBPH_MAX_DIST`  | `120`             | Usado para converter distância LBPH → porcentagem  |
| Confiança mínima | 50%               | Abaixo disso, o rosto é considerado "Desconhecido" |

### ⚖️ Nota Ética sobre Uso de Dados Faciais
#### O uso de sistemas de reconhecimento facial envolve riscos e responsabilidades.
#### Recomendações importantes:
- Utilize este projeto apenas para fins educacionais ou pessoais, nunca para vigilância, controle ou identificação sem consentimento.
- Sempre obtenha permissão explícita de qualquer pessoa cujo rosto será capturado, armazenado ou processado.
- Armazene e trate os dados faciais com cuidado:
    > Não compartilhe pastas de dataset <br>
    Evite sincronizar imagens sensíveis na nuvem <br>
    Exclua dados quando o usuário solicitar

#### Lembre-se: reconhecimento facial pode gerar erros, vieses e falsos positivos, e não deve ser usado como critério único para decisões importantes. O respeito à privacidade e ao consentimento é essencial em qualquer aplicação com biometria.

## Integrantes
| Nome                          | RM       |
|-------------------------------|----------|
| Erick Molina                  | 553852   |
| Felipe Castro Salazar        | 553464   |
| Marcelo Vieira de Melo       | 552953   |
| Rayara Amaro Figueiredo      | 552635   |
| Victor Rodrigues             | 554158   |
