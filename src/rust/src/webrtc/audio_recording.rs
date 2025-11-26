//
// Copyright 2025 Signal Messenger, LLC
// SPDX-License-Identifier: AGPL-3.0-only
//

//! Audio recording infrastructure using WebRTC AudioTransport interface.
//!
//! This module provides the ability to capture audio samples from the
//! AudioTransport interface and convert them to MediaStream objects for
//! call recording purposes.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Trait for audio recording sinks that capture audio samples
/// from the AudioTransport interface.
pub trait AudioRecordingSink: Send + Sync {
    /// Called when local audio samples are available.
    ///
    /// # Arguments
    /// * `samples` - Audio samples (i16 format)
    /// * `sample_rate` - Sample rate in Hz (typically 48000)
    /// * `channels` - Number of audio channels (typically 1 for mono)
    /// * `timestamp` - Timestamp of the audio samples
    fn on_local_audio_samples(
        &self,
        samples: &[i16],
        sample_rate: u32,
        channels: u32,
        timestamp: Duration,
    );

    /// Called when remote audio samples are available.
    ///
    /// # Arguments
    /// * `samples` - Audio samples (i16 format)
    /// * `sample_rate` - Sample rate in Hz (typically 48000)
    /// * `channels` - Number of audio channels (typically 2 for stereo)
    /// * `timestamp` - Timestamp of the audio samples
    fn on_remote_audio_samples(
        &self,
        samples: &[i16],
        sample_rate: u32,
        channels: u32,
        timestamp: Duration,
    );
}

/// Audio chunk with metadata for buffering and synchronization.
#[derive(Clone, Debug)]
struct AudioChunk {
    samples: Vec<i16>,
    timestamp: Duration,
    sample_rate: u32,
    channels: u32,
}

/// Recording sink that buffers audio samples for conversion to MediaStream.
///
/// This sink collects audio samples from both local and remote sources,
/// buffers them, and provides methods to convert them to MediaStream format.
pub struct MediaStreamAudioSink {
    local_buffer: Arc<Mutex<VecDeque<AudioChunk>>>,
    remote_buffer: Arc<Mutex<VecDeque<AudioChunk>>>,
    sample_rate: u32,
    local_channels: u32,
    remote_channels: u32,
    max_buffer_size: usize,
    is_active: Arc<Mutex<bool>>,
}

impl MediaStreamAudioSink {
    /// Create a new MediaStreamAudioSink.
    ///
    /// # Arguments
    /// * `sample_rate` - Expected sample rate (typically 48000 Hz)
    pub fn new(sample_rate: u32) -> Self {
        Self {
            local_buffer: Arc::new(Mutex::new(VecDeque::new())),
            remote_buffer: Arc::new(Mutex::new(VecDeque::new())),
            sample_rate,
            local_channels: 1, // Mono for local
            remote_channels: 2, // Stereo for remote
            max_buffer_size: 48000, // 1 second at 48kHz
            is_active: Arc::new(Mutex::new(false)),
        }
    }

    /// Set the sink as active (recording).
    pub fn set_active(&self, active: bool) {
        if let Ok(mut state) = self.is_active.lock() {
            *state = active;
            if !active {
                // Clear buffers when stopping
                if let Ok(mut local) = self.local_buffer.lock() {
                    local.clear();
                }
                if let Ok(mut remote) = self.remote_buffer.lock() {
                    remote.clear();
                }
            }
        }
    }

    /// Check if the sink is active.
    pub fn is_active(&self) -> bool {
        self.is_active.lock().map(|s| *s).unwrap_or(false)
    }

    /// Get buffered local audio chunks.
    pub fn get_local_chunks(&self) -> Vec<AudioChunk> {
        self.local_buffer
            .lock()
            .map(|buf| buf.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get buffered remote audio chunks.
    pub fn get_remote_chunks(&self) -> Vec<AudioChunk> {
        self.remote_buffer
            .lock()
            .map(|buf| buf.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get local channel count.
    pub fn local_channels(&self) -> u32 {
        self.local_channels
    }

    /// Get remote channel count.
    pub fn remote_channels(&self) -> u32 {
        self.remote_channels
    }
}

impl AudioRecordingSink for MediaStreamAudioSink {
    fn on_local_audio_samples(
        &self,
        samples: &[i16],
        sample_rate: u32,
        channels: u32,
        timestamp: Duration,
    ) {
        // Only buffer if active
        if !self.is_active() {
            return;
        }

        if let Ok(mut buffer) = self.local_buffer.lock() {
            buffer.push_back(AudioChunk {
                samples: samples.to_vec(),
                timestamp,
                sample_rate,
                channels,
            });

            // Limit buffer size to prevent memory issues
            while buffer.len() > self.max_buffer_size {
                buffer.pop_front();
            }
        }
    }

    fn on_remote_audio_samples(
        &self,
        samples: &[i16],
        sample_rate: u32,
        channels: u32,
        timestamp: Duration,
    ) {
        // Only buffer if active
        if !self.is_active() {
            return;
        }

        if let Ok(mut buffer) = self.remote_buffer.lock() {
            buffer.push_back(AudioChunk {
                samples: samples.to_vec(),
                timestamp,
                sample_rate,
                channels,
            });

            // Limit buffer size to prevent memory issues
            while buffer.len() > self.max_buffer_size {
                buffer.pop_front();
            }
        }
    }
}

