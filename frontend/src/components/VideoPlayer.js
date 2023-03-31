import React, { useRef, useState, useEffect } from 'react';

const VideoPlayer = ({ videoSrc }) => {
  const videoRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    const video = videoRef.current;

    const handlePlay = () => {
      setIsPlaying(true);
    };

    const handlePause = () => {
      setIsPlaying(false);
    };

    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);

    return () => {
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
    };
  }, []);

  const handlePlayPause = () => {
    const video = videoRef.current;

    if (isPlaying) {
      video.pause();
    } else {
      video.play();
    }
  };

  return (
    <div className="video-player">
      <video src={videoSrc} ref={videoRef} onClick={handlePlayPause} />
      <button onClick={handlePlayPause}>
        {isPlaying ? 'Pause' : 'Play'}
      </button>
    </div>
  );
};

export default VideoPlayer;
