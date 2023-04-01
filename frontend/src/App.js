import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [words, setWords] = useState([]);
  const [selectedWords, setSelectedWords] = useState([]);
  const [videoUrl, setVideoUrl] = useState(null);

  const handleFileInputChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    setProcessing(true);
    const formData = new FormData();
    formData.append('video', selectedFile);
    axios.post('/process_video', formData)
      .then(response => {
        setWords(response.data.words);
        setVideoUrl(response.data.video_url);
        setProcessing(false);
      })
      .catch(error => {
        console.error(error);
        setProcessing(false);
      });
  };

  const handleWordSelection = (word) => {
    const index = selectedWords.indexOf(word);
    if (index === -1) {
      setSelectedWords([...selectedWords, word]);
    } else {
      setSelectedWords(selectedWords.filter(w => w !== word));
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <label htmlFor="video">Select video:</label>
        <input type="file" id="video" name="video" onChange={handleFileInputChange} />
        <br /><br />
        <button type="submit" disabled={!selectedFile || processing}>Submit</button>
      </form>
      {processing && <p>Processing video...</p>}
      {videoUrl && <video controls src={videoUrl} />}
      {words.length > 0 && (
        <div>
          <h2>Word statistics:</h2>
          <ul>
            {words.map(word => (
              <li key={word.word}>
                {word.word}: {word.count}
                <button onClick={() => handleWordSelection(word.word)}>
                  {selectedWords.includes(word.word) ? 'Deselect' : 'Select'}
                </button>
              </li>
            ))}
          </ul>
          <button disabled={selectedWords.length === 0}>
            Mute selected words and start streaming
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
