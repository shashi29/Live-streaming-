import axios from 'axios';

const BASE_URL = 'http://localhost:8000'; // replace with your backend server URL

export const uploadVideo = async (formData) => {
  try {
    const response = await axios.post(`${BASE_URL}/api/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  } catch (error) {
    console.error(error);
    throw new Error('Error uploading video');
  }
};

export const getWordStatistics = async (videoId) => {
  try {
    const response = await axios.get(`${BASE_URL}/api/word-statistics/${videoId}`);
    return response.data;
  } catch (error) {
    console.error(error);
    throw new Error('Error getting word statistics');
  }
};

export const muteVideoWords = async (videoId, words) => {
  try {
    const response = await axios.post(`${BASE_URL}/api/mute-words/${videoId}`, { words });
    return response.data;
  } catch (error) {
    console.error(error);
    throw new Error('Error muting video words');
  }
};
