from flask import Flask, request, jsonify
import json
import uuid
from datetime import datetime

app = Flask(__name__)

# In-memory storage for leave records
leave_records = []

# Helper function to calculate days between dates
def calculate_days(start_date, end_date):
    from datetime import datetime
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    return (end - start).days + 1

# Helper function to validate date format
def is_valid_date(date_str):
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

# Helper function to validate leave type
def is_valid_leave_type(leave_type):
    valid_types = ['annual_leave', 'sick_leave', 'personal_leave', 
                   'marriage_leave', 'maternity_leave', 'paternity_leave', 'bereavement_leave']
    return leave_type in valid_types

# Helper function to validate status
def is_valid_status(status):
    return status in ['pending', 'approved', 'rejected']

# GET /leaves - Retrieve all leave records with optional filtering
@app.route('/api/leaves', methods=['GET'])
def get_leaves():
    try:
        # Get query parameters
        employee_name = request.args.get('employee_name')
        leave_type = request.args.get('leave_type')
        status = request.args.get('status')
        
        # Filter records based on query parameters
        filtered_records = leave_records.copy()
        
        if employee_name:
            filtered_records = [record for record in filtered_records 
                              if record['employee_name'].lower() == employee_name.lower()]
        
        if leave_type:
            if not is_valid_leave_type(leave_type):
                return jsonify({'detail': 'Invalid leave_type'}), 400
            filtered_records = [record for record in filtered_records 
                              if record['leave_type'] == leave_type]
        
        if status:
            if not is_valid_status(status):
                return jsonify({'detail': 'Invalid status'}), 400
            filtered_records = [record for record in filtered_records 
                              if record['status'] == status]
        
        return jsonify(filtered_records), 200
        
    except Exception as e:
        return jsonify({'detail': 'Server error'}), 500

# POST /leaves - Create a new leave record
@app.route('/api/leaves', methods=['POST'])
def create_leave():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['employee_id', 'employee_name', 'leave_type', 'start_date', 'end_date', 'reason']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'detail': f'Missing required field: {field}'}), 400
        
        # Validate date format
        if not is_valid_date(data['start_date']):
            return jsonify({'detail': 'Invalid start_date format. Use YYYY-MM-DD'}), 400
        if not is_valid_date(data['end_date']):
            return jsonify({'detail': 'Invalid end_date format. Use YYYY-MM-DD'}), 400
        
        # Validate date logic
        if data['end_date'] < data['start_date']:
            return jsonify({'detail': 'end_date must be greater than or equal to start_date'}), 400
        
        # Validate leave_type
        if not is_valid_leave_type(data['leave_type']):
            return jsonify({'detail': 'Invalid leave_type'}), 400
        
        # Validate reason length
        if len(data['reason']) > 200:
            return jsonify({'detail': 'reason must be less than or equal to 200 characters'}), 422
        
        # Calculate days
        days = calculate_days(data['start_date'], data['end_date'])
        
        # Create new leave record
        new_leave = {
            'id': str(uuid.uuid4()),
            'employee_id': data['employee_id'],
            'employee_name': data['employee_name'],
            'leave_type': data['leave_type'],
            'start_date': data['start_date'],
            'end_date': data['end_date'],
            'days': days,
            'apply_time': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'status': 'pending',
            'approver': '',
            'reason': data['reason']
        }
        
        leave_records.append(new_leave)
        
        return jsonify(new_leave), 201
        
    except Exception as e:
        return jsonify({'detail': 'Server error'}), 500

# GET /leaves/{id} - Retrieve a specific leave record by ID
@app.route('/api/leaves/<id>', methods=['GET'])
def get_leave_by_id(id):
    try:
        leave = next((record for record in leave_records if record['id'] == id), None)
        
        if not leave:
            return jsonify({'detail': 'Leave record not found'}), 404
        
        return jsonify(leave), 200
        
    except Exception as e:
        return jsonify({'detail': 'Server error'}), 500

# PUT /leaves/{id} - Update an existing leave record
@app.route('/api/leaves/<id>', methods=['PUT'])
def update_leave(id):
    try:
        data = request.get_json()
        
        # Find the leave record
        leave = next((record for record in leave_records if record['id'] == id), None)
        
        if not leave:
            return jsonify({'detail': 'Leave record not found'}), 404
        
        # Check if status is pending
        if leave['status'] != 'pending':
            return jsonify({'detail': 'Cannot update record with status other than "pending"'}), 403
        
        # Validate required fields
        required_fields = ['employee_id', 'employee_name', 'leave_type', 'start_date', 'end_date', 'reason']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'detail': f'Missing required field: {field}'}), 400
        
        # Validate date format
        if not is_valid_date(data['start_date']):
            return jsonify({'detail': 'Invalid start_date format. Use YYYY-MM-DD'}), 400
        if not is_valid_date(data['end_date']):
            return jsonify({'detail': 'Invalid end_date format. Use YYYY-MM-DD'}), 400
        
        # Validate date logic
        if data['end_date'] < data['start_date']:
            return jsonify({'detail': 'end_date must be greater than or equal to start_date'}), 400
        
        # Validate leave_type
        if not is_valid_leave_type(data['leave_type']):
            return jsonify({'detail': 'Invalid leave_type'}), 400
        
        # Validate reason length
        if len(data['reason']) > 200:
            return jsonify({'detail': 'reason must be less than or equal to 200 characters'}), 422
        
        # Update the record
        leave['employee_id'] = data['employee_id']
        leave['employee_name'] = data['employee_name']
        leave['leave_type'] = data['leave_type']
        leave['start_date'] = data['start_date']
        leave['end_date'] = data['end_date']
        leave['days'] = calculate_days(data['start_date'], data['end_date'])
        leave['reason'] = data['reason']
        
        return jsonify(leave), 200
        
    except Exception as e:
        return jsonify({'detail': 'Server error'}), 500

# DELETE /leaves/{id} - Delete a leave record
@app.route('/api/leaves/<id>', methods=['DELETE'])
def delete_leave(id):
    try:
        leave = next((record for record in leave_records if record['id'] == id), None)
        
        if not leave:
            return jsonify({'detail': 'Leave record not found'}), 404
        
        # Check if status is pending
        if leave['status'] != 'pending':
            return jsonify({'detail': 'Cannot delete record with status other than "pending"'}), 403
        
        leave_records.remove(leave)
        
        return '', 204
        
    except Exception as e:
        return jsonify({'detail': 'Server error'}), 500

# PATCH /leaves/{id}/status - Update the status of a leave record
@app.route('/api/leaves/<id>/status', methods=['PATCH'])
def update_leave_status(id):
    try:
        data = request.get_json()
        
        # Validate required fields
        if 'status' not in data:
            return jsonify({'detail': 'Missing required field: status'}), 400
        
        # Find the leave record
        leave = next((record for record in leave_records if record['id'] == id), None)
        
        if not leave:
            return jsonify({'detail': 'Leave record not found'}), 404
        
        # Check if current status is pending
        if leave['status'] != 'pending':
            return jsonify({'detail': 'Cannot update status of record that is not "pending"'}), 403
        
        # Validate status value
        if data['status'] not in ['approved', 'rejected']:
            return jsonify({'detail': 'Invalid status value. Must be "approved" or "rejected"'}), 400
        
        # Update status and approver
        leave['status'] = data['status']
        leave['approver'] = data.get('approver', '')
        
        return jsonify(leave), 200
        
    except Exception as e:
        return jsonify({'detail': 'Server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
