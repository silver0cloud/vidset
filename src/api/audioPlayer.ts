/**
 * audioPlayer.ts
 * --------------
 * Plays streamed PCM16 audio chunks (from /ws/speak) in real time
 * using the Web Audio API, queuing chunks for gapless playback.
 */

const SAMPLE_RATE = 22050;

export class StreamingAudioPlayer {
  private ctx: AudioContext;
  private nextStartTime = 0;
  private playing = false;

  constructor() {
    this.ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
  }

  /** Feed a raw PCM16 chunk (ArrayBuffer) for immediate scheduled playback. */
  pushChunk(buffer: ArrayBuffer): void {
    const pcm16 = new Int16Array(buffer);
    const float32 = new Float32Array(pcm16.length);
    for (let i = 0; i < pcm16.length; i++) {
      float32[i] = pcm16[i] / 32768;
    }

    const audioBuffer = this.ctx.createBuffer(1, float32.length, SAMPLE_RATE);
    audioBuffer.copyToChannel(float32, 0);

    const source = this.ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.ctx.destination);

    const now = this.ctx.currentTime;
    const startAt = Math.max(now, this.nextStartTime);
    source.start(startAt);

    this.nextStartTime = startAt + audioBuffer.duration;
    this.playing = true;
  }

  /** Call when generation is done — no more chunks coming. */
  finish(): void {
    this.playing = false;
  }

  stop(): void {
    this.ctx.close();
    this.playing = false;
  }

  isPlaying(): boolean {
    return this.playing;
  }
}
