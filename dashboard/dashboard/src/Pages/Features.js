import React from 'react';

export default function Features() {

    const buttonContainerStyle = {
        display: 'flex',
        flexDirection: 'column', // Display buttons in a column
        alignItems: 'center',
    };

    const buttonStyle = {
        margin: '10px',
        padding: '10px 20px',
        backgroundColor: 'blue',
        color: 'white',
        border: 'none',
        borderRadius: '5px',
        cursor: 'pointer',
    };

    return (
        <div>
            <div style={buttonContainerStyle}>
                <div>
                    <button style={buttonStyle}>Object Finder</button>
                </div>
                <div>
                    <button style={buttonStyle}>Colour Detection</button>
                </div>
                <div>
                    <button style={buttonStyle}>OCR</button>
                </div>
                <div>
                    <button style={buttonStyle}>Navigation</button>
                </div>
                <div>
                    <button style={buttonStyle}>Environment Detection</button>
                </div>
            </div>
        </div>
    );
}
