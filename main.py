import cv2
import os
import numpy as np

# Diretório do dataset
DATASET_PATH = "faces_dataset"
os.makedirs(DATASET_PATH, exist_ok=True)

# Haar Cascade para detecção
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Quantidade de imagens para treinamento (optamos por 150)
CAPTURE_COUNT = 150

# Limite máximo estimado para converter distância LBPH → 0–100% (Basicamente define o nível de confiança)
LBPH_MAX_DIST = 120


# ============================================================
# CAPTURA DE FOTOS DO ROSTO
# ============================================================

def capturar_faces(nome):
    pessoa_path = os.path.join(DATASET_PATH, nome)

    if os.path.exists(pessoa_path):
        print("[ERRO] Usuário já cadastrado.") # Validando se usuário já está cadastrado
        return False

    os.makedirs(pessoa_path, exist_ok=True)

    cam = cv2.VideoCapture(0)
    count = 0
    print(f"\n[INFO] Capturando {CAPTURE_COUNT} imagens...")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            cv2.imwrite(f"{pessoa_path}/{count}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"Capturando {count}/{CAPTURE_COUNT}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == 27 or count >= CAPTURE_COUNT:
            break

    cam.release()
    cv2.destroyAllWindows()

    print("[INFO] Captura concluída.\n")
    return True


# ============================================================
# TREINAMENTO DO LBPH
# ============================================================

def treinar_reconhecedor():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = [], []
    label_map = {}

    print("[INFO] Treinando modelo...")

    nomes = os.listdir(DATASET_PATH)

    for idx, nome in enumerate(nomes):
        pessoa_path = os.path.join(DATASET_PATH, nome)
        label_map[idx] = nome

        for img_name in os.listdir(pessoa_path):
            img_path = os.path.join(pessoa_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            faces.append(img)
            ids.append(idx)

    if len(faces) == 0:
        print("[ERRO] Nenhuma imagem válida encontrada.")
        return None, None

    ids = np.array(ids, dtype=np.int32)
    recognizer.train(faces, ids)

    print("[INFO] Modelo treinado com sucesso!\n")
    return recognizer, label_map


# ============================================================
# CONVERTER DISTÂNCIA LBPH → PORCENTAGEM (0 a 100)
# ============================================================

def converter_confianca(dist):
    conf = 100 - (dist / LBPH_MAX_DIST * 100)
    return max(0, min(100, conf))


# ============================================================
# RECONHECIMENTO AO VIVO — MULTIPLOS ROSTOS
# ============================================================

def reconhecer_faces(recognizer, label_map):
    cam = cv2.VideoCapture(0)
    print("[INFO] Reconhecimento iniciado. (ESC para sair)")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop para múltiplas pessoas
        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]

            label, distancia = recognizer.predict(face_img)
            confianca = converter_confianca(distancia)

            if confianca >= 50:
                nome = label_map[label]
                cor = (0, 255, 0)
                texto = f"{nome} ({confianca:.1f}%)"
            else:
                nome = "Desconhecido"
                cor = (0, 0, 255)
                texto = f"{nome}"

            cv2.putText(frame, texto, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


# ============================================================
# LOOP PRINCIPAL — CÂMERA INICIA RECONHECENDO AUTOMATICAMENTE
# ============================================================

def main():
    # Se já existem rostos, treina automaticamente e inicia reconhecimento facial
    if len(os.listdir(DATASET_PATH)) > 0:
        print("[INFO] Rostos já cadastrados. Treinando modelo...")
        recognizer, label_map = treinar_reconhecedor()
        if recognizer:
            reconhecer_faces(recognizer, label_map)
        return

    # Caso não existam rostos cadastrados, o sistema oferece cadastro
    cam = cv2.VideoCapture(0)
    print("[INFO] Nenhum rosto cadastrado. Aperte C para cadastrar.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        cv2.putText(frame, "Aperte C para cadastrar seu rosto",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            cam.release()
            cv2.destroyAllWindows()

            nome = input("Digite seu nome: ").strip()
            if capturar_faces(nome):
                recognizer, label_map = treinar_reconhecedor()
                if recognizer:
                    reconhecer_faces(recognizer, label_map)
            return

        if key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


main()
