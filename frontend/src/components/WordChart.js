import React, { useState, useEffect } from 'react';
import Chart from 'chart.js';

const WordChart = ({ words }) => {
  const [chart, setChart] = useState(null);

  useEffect(() => {
    // Create a new chart with the given data
    const ctx = document.getElementById('wordChart');
    const chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: Object.keys(words),
        datasets: [{
          label: 'Frequency',
          data: Object.values(words),
          backgroundColor: '#007bff',
        }],
      },
      options: {
        scales: {
          yAxes: [{
            ticks: {
              beginAtZero: true,
              stepSize: 1,
            },
          }],
        },
      },
    });

    // Save the chart to the state
    setChart(chart);

    // Cleanup when the component unmounts
    return () => {
      chart.destroy();
    };
  }, [words]);

  return (
    <div>
      <canvas id="wordChart" width="400" height="400"></canvas>
    </div>
  );
};

export default WordChart;
