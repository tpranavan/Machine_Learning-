import csv

# Read the CSV file and extract unique values from the "Interested_Area" column
unique_values = set()
with open('../Career_dataset.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        interested_area = row['Interested_Area']
        if interested_area:
            unique_values.add(interested_area)

# Generate HTML code with the unique values for the <option> elements
options_html = ""
for value in unique_values:
    options_html += f'<option value="{value}">{value}</option>\n'

# Update the <select> element with the generated options_html
input_html = '''
<div class="col-lg-6">
    <fieldset>
        <label for="interested_area">Interested Area:</label>
        <select name="interested_area" id="interested_area" required>
            <option value="">Select Interested Area</option>
            {options_html}
        </select>
    </fieldset>
</div>
'''

input_html = input_html.format(options_html=options_html)

# Print the updated HTML code
print(input_html)
