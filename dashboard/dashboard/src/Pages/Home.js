import React from 'react';

export default function Home() {
    const logoStyle = {
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '60vh', // Reduce the height to occupy only 60% of the viewport height
    };

    const buttonContainerStyle = {
        display: 'flex',
        flexDirection: 'column', // Display buttons in a column
        alignItems: 'center',
        marginTop: '50px', // Add margin-top to create space below the image
    };

    const buttonStyle = {
        margin: '10px',
        padding: '10px 20px',
        backgroundImage: 'linear-gradient(to left, #FA52FF, #6A11CB)', // Apply the linear gradient background
        color: 'white',
        border: 'none',
        borderRadius: '5px',
        cursor: 'pointer',
        fontSize: '25px'
    };

    return (
        <div>
            <div style={logoStyle}>
                <img
                    src="/weifinder_logo.png" // Replace with the path to your logo image
                    alt="Logo"
                    style={{ maxWidth: '100%', maxHeight: '90%' }}
                />
            </div>
            <div style={buttonContainerStyle}>
                <div>
                    <button style={buttonStyle}>Start WeiFinder</button>
                </div>
            </div>
        </div>
    );
}
