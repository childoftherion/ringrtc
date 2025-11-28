//
// Copyright 2025 Signal Messenger, LLC
// SPDX-License-Identifier: AGPL-3.0-only
//

//! Video recording infrastructure using WebRTC VideoSink interface.
//!
//! This module provides the ability to capture video frames from both
//! local (outgoing) and remote (incoming) video streams and buffer
//! them for conversion to MediaStream objects for call recording.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::lite::sfu::DemuxId;
use crate::webrtc::media::{VideoFrame, VideoPixelFormat, VideoSink};

/// Trait for video recording sinks that capture video frames
/// from the VideoSink interface.
pub trait VideoRecordingSink: Send + Sync {
    /// Called when a local (outgoing) video frame is available.
    ///
    /// # Arguments
    /// * `frame` - Video frame data
    fn on_local_video_frame(&self, frame: VideoFrame);

    /// Called when a remote (incoming) video frame is available.
    ///
    /// # Arguments
    /// * `demux_id` - Demux ID identifying the remote participant
    /// * `frame` - Video frame data
    fn on_remote_video_frame(&self, demux_id: DemuxId, frame: VideoFrame);
}

/// Video chunk with metadata for buffering and synchronization.
#[derive(Clone)]
pub struct VideoChunk {
    pub width: u32,
    pub height: u32,
    pub pixel_format: VideoPixelFormat,
    pub buffer: Vec<u8>,
    pub timestamp: Duration,
    pub demux_id: Option<u32>, // None for local, Some(demux_id) for remote
}

/// Recording sink that buffers video frames for conversion to MediaStream.
///
/// This sink collects video frames from both local and remote sources,
/// buffers them, and provides methods to convert them to MediaStream format.
pub struct MediaStreamVideoSink {
    local_buffer: Arc<Mutex<VecDeque<VideoChunk>>>,
    remote_buffer: Arc<Mutex<VecDeque<VideoChunk>>>,
    is_active: Arc<Mutex<bool>>,
    max_buffer_size: usize, // Maximum number of frames to buffer
}

impl MediaStreamVideoSink {
    /// Create a new MediaStreamVideoSink.
    ///
    /// # Arguments
    /// * `max_buffer_size` - Maximum number of frames to buffer (default: 300 for 10 seconds at 30fps)
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            local_buffer: Arc::new(Mutex::new(VecDeque::new())),
            remote_buffer: Arc::new(Mutex::new(VecDeque::new())),
            is_active: Arc::new(Mutex::new(false)),
            max_buffer_size,
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

    /// Get buffered local video chunks (copies them).
    pub fn get_local_chunks(&self) -> Vec<VideoChunk> {
        self.local_buffer
            .lock()
            .map(|buf| buf.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get buffered remote video chunks (copies them).
    pub fn get_remote_chunks(&self) -> Vec<VideoChunk> {
        self.remote_buffer
            .lock()
            .map(|buf| buf.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Drain local video chunks (removes and returns them).
    /// This is used for real-time streaming to avoid buffer overflow.
    pub fn drain_local_chunks(&self) -> Vec<VideoChunk> {
        self.local_buffer
            .lock()
            .map(|mut buf| {
                let mut chunks = Vec::new();
                while let Some(chunk) = buf.pop_front() {
                    chunks.push(chunk);
                }
                chunks
            })
            .unwrap_or_default()
    }

    /// Drain remote video chunks (removes and returns them).
    /// This is used for real-time streaming to avoid buffer overflow.
    pub fn drain_remote_chunks(&self) -> Vec<VideoChunk> {
        self.remote_buffer
            .lock()
            .map(|mut buf| {
                let mut chunks = Vec::new();
                while let Some(chunk) = buf.pop_front() {
                    chunks.push(chunk);
                }
                chunks
            })
            .unwrap_or_default()
    }

    /// Get the latest local video chunk without removing it.
    pub fn peek_latest_local_chunk(&self) -> Option<VideoChunk> {
        self.local_buffer
            .lock()
            .ok()
            .and_then(|buf| buf.back().cloned())
    }

    /// Get the latest remote video chunk without removing it.
    pub fn peek_latest_remote_chunk(&self) -> Option<VideoChunk> {
        self.remote_buffer
            .lock()
            .ok()
            .and_then(|buf| buf.back().cloned())
    }
}

impl VideoRecordingSink for MediaStreamVideoSink {
    fn on_local_video_frame(&self, frame: VideoFrame) {
        // Only buffer if active
        if !self.is_active() {
            return;
        }

        // Convert frame to chunk
        let width = frame.width();
        let height = frame.height();
        let pixel_format = VideoPixelFormat::Rgba; // We convert to RGBA

        // Allocate buffer for RGBA data (4 bytes per pixel)
        let mut buffer = vec![0u8; (width * height * 4) as usize];
        
        // Convert frame to RGBA
        frame.to_rgba(&mut buffer);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();

        let chunk = VideoChunk {
            width,
            height,
            pixel_format,
            buffer,
            timestamp,
            demux_id: None, // Local frames don't have demux_id
        };

        if let Ok(mut buf) = self.local_buffer.lock() {
            buf.push_back(chunk);

            // Limit buffer size to prevent memory issues
            while buf.len() > self.max_buffer_size {
                buf.pop_front();
            }
        }
    }

    fn on_remote_video_frame(&self, demux_id: DemuxId, frame: VideoFrame) {
        // Only buffer if active
        if !self.is_active() {
            return;
        }

        // Convert frame to chunk
        let width = frame.width();
        let height = frame.height();
        let pixel_format = VideoPixelFormat::Rgba; // We convert to RGBA

        // Allocate buffer for RGBA data (4 bytes per pixel)
        let mut buffer = vec![0u8; (width * height * 4) as usize];
        
        // Convert frame to RGBA
        frame.to_rgba(&mut buffer);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();

        let chunk = VideoChunk {
            width,
            height,
            pixel_format,
            buffer,
            timestamp,
            demux_id: Some(demux_id),
        };

        if let Ok(mut buf) = self.remote_buffer.lock() {
            buf.push_back(chunk);

            // Limit buffer size to prevent memory issues
            while buf.len() > self.max_buffer_size {
                buf.pop_front();
            }
        }
    }
}

/// VideoSink implementation that forwards frames to recording sinks.
/// This wraps the existing VideoSink and adds recording capability.
pub struct RecordingVideoSink {
    inner: Box<dyn VideoSink>,
    recording_sinks: Arc<Mutex<Vec<Arc<dyn VideoRecordingSink>>>>,
}

impl RecordingVideoSink {
    pub fn new(inner: Box<dyn VideoSink>) -> Self {
        Self {
            inner,
            recording_sinks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn add_recording_sink(&self, sink: Arc<dyn VideoRecordingSink>) {
        if let Ok(mut sinks) = self.recording_sinks.lock() {
            sinks.push(sink);
        }
    }

    pub fn remove_recording_sink(&self, sink: Arc<dyn VideoRecordingSink>) {
        if let Ok(mut sinks) = self.recording_sinks.lock() {
            sinks.retain(|s| !Arc::ptr_eq(s, sink));
        }
    }
}

impl VideoSink for RecordingVideoSink {
    fn on_video_frame(&self, demux_id: DemuxId, frame: VideoFrame) {
        // Forward to inner sink first
        self.inner.on_video_frame(demux_id, frame.clone());

        // Then forward to recording sinks (remote frames)
        if let Ok(sinks) = self.recording_sinks.lock() {
            for sink in sinks.iter() {
                sink.on_remote_video_frame(demux_id, frame.clone());
            }
        }
    }

    fn box_clone(&self) -> Box<dyn VideoSink> {
        Box::new(RecordingVideoSink {
            inner: self.inner.box_clone(),
            recording_sinks: Arc::clone(&self.recording_sinks),
        })
    }
}

