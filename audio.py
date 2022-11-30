import pyaudio,wave
import tqdm
def read_audio():
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = 1

        # 实例化一个PyAudio对象
        pa = pyaudio.PyAudio()
        # 打开声卡，设置 采样深度为16位、声道数为2、采样率为16、输入、采样点缓存数量为2048
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
        # 新建一个列表，用来存储采样到的数据
        record_buf = []
        count = 0
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            audio_data = stream.read(2048)  # 读出声卡缓冲区的音频数据
            record_buf.append(audio_data)  # 将读出的音频数据追加到record_buf列表
            count += 1
            print('*')
        #wf = wave.open('quake_sounds/test/audio-0-0.wav', 'wb')  # 创建一个音频文件，名字为“01.wav"
        #wf.setnchannels(1)  # 设置声道数为1
        #wf.setsampwidth(2)  # 设置采样深度为2
        #wf.setframerate(RATE)
        # 将数据写入创建的音频文件
        #wf.writeframes("".encode().join(record_buf))
        # 写完后将文件关闭
        #wf.close()
        # 停止声卡
        stream.stop_stream()
        # 关闭声卡
        stream.close()
while True:
    read_audio()
