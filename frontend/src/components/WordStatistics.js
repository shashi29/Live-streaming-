import React from "react";

const WordStatistics = ({ data }) => {
  const totalWords = Object.keys(data).reduce((acc, curr) => acc + data[curr], 0);
  const uniqueWords = Object.keys(data).length;
  const wordOccurrences = Object.entries(data).sort((a, b) => b[1] - a[1]);

  return (
    <div>
      <h3>Statistics</h3>
      <p>Total words: {totalWords}</p>
      <p>Unique words: {uniqueWords}</p>
      <div>
        <table>
          <thead>
            <tr>
              <th>Word</th>
              <th>Count</th>
            </tr>
          </thead>
          <tbody>
            {wordOccurrences.map(([word, count]) => (
              <tr key={word}>
                <td>{word}</td>
                <td>{count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default WordStatistics;
