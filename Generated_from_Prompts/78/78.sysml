package 'TransportationUseCase078' {
	part def Vehicle;
	part def Person;
	
	use case def 'Provide Transportation' {
		subject vehicle : Vehicle;
		actor driver : Person;
		actor passengers : Person[0..4];
	}
}
